from __future__ import annotations
import abc
import copy
import os
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Literal, Union
import numpy as np
from monty.design_patterns import cached_class
from monty.json import MSONable
from monty.serialization import loadfn
from tqdm import tqdm
from uncertainties import ufloat
from pymatgen.analysis.structure_analyzer import oxide_type, sulfide_type
from pymatgen.core import SETTINGS, Composition, Element
from pymatgen.entries.computed_entries import (
from pymatgen.io.vasp.sets import MITRelaxSet, MPRelaxSet, VaspInputSet
from pymatgen.util.due import Doi, due
@cached_class
class MaterialsProject2020Compatibility(Compatibility):
    """This class implements the Materials Project 2020 energy correction scheme, which
    incorporates uncertainty quantification and allows for mixing of GGA and GGA+U entries
    (see References).

    Note that this scheme should only be applied to VASP calculations that use the
    Materials Project input set parameters (see pymatgen.io.vasp.sets.MPRelaxSet). Using
    this compatibility scheme on calculations with different parameters is not valid.

    Note: While the correction scheme is largely composition-based, the energy corrections
    applied to ComputedEntry and ComputedStructureEntry can differ for O and S-containing
    structures if entry.data['oxidation_states'] is not populated or explicitly set. This
    occurs because pymatgen will use atomic distances to classify O and S anions as
    superoxide/peroxide/oxide and sulfide/polysulfide, resp. when oxidation states are not
    provided. If you want the most accurate corrections possible, supply pre-defined
    oxidation states to entry.data or pass ComputedStructureEntry.
    """

    def __init__(self, compat_type: str='Advanced', correct_peroxide: bool=True, check_potcar: bool=True, check_potcar_hash: bool=False, config_file: str | None=None) -> None:
        """
        Args:
            compat_type: Two options, GGA or Advanced. GGA means all GGA+U
                entries are excluded. Advanced means the GGA/GGA+U mixing scheme
                of Jain et al. (see References) is implemented. In this case,
                entries which are supposed to be calculated in GGA+U (i.e.,
                transition metal oxides and fluorides) will have the corresponding
                GGA entries excluded. For example, Fe oxides should
                have a U value under the Advanced scheme. An Fe oxide run in GGA
                will therefore be excluded.

                To use the "Advanced" type, Entry.parameters must contain a "hubbards"
                key which is a dict of all non-zero Hubbard U values used in the
                calculation. For example, if you ran a Fe2O3 calculation with
                Materials Project parameters, this would look like
                entry.parameters["hubbards"] = {"Fe": 5.3}. If the "hubbards" key
                is missing, a GGA run is assumed. Entries obtained from the
                MaterialsProject database will automatically have these fields
                populated. Default: "Advanced"
            correct_peroxide: Specify whether peroxide/superoxide/ozonide
                corrections are to be applied or not. If false, all oxygen-containing
                compounds are assigned the 'oxide' correction. Default: True
            check_potcar (bool): Check that the POTCARs used in the calculation are consistent
                with the Materials Project parameters. False bypasses this check altogether. Default: True
                Can also be disabled globally by running `pmg config --add PMG_POTCAR_CHECKS false`.
            check_potcar_hash (bool): Use potcar hash to verify POTCAR settings are
                consistent with MPRelaxSet. If False, only the POTCAR symbols will
                be used. Default: False
            config_file (Path): Path to the selected compatibility.yaml config file.
                If None, defaults to `MP2020Compatibility.yaml` distributed with
                pymatgen.

        References:
            Wang, A., Kingsbury, R., McDermott, M., Horton, M., Jain. A., Ong, S.P.,
                Dwaraknath, S., Persson, K. A framework for quantifying uncertainty
                in DFT energy corrections. Scientific Reports 11: 15496, 2021.
                https://doi.org/10.1038/s41598-021-94550-5

            Jain, A. et al. Formation enthalpies by mixing GGA and GGA + U calculations.
                Phys. Rev. B - Condens. Matter Mater. Phys. 84, 1-10 (2011).
        """
        if compat_type not in ['GGA', 'Advanced']:
            raise CompatibilityError(f'Invalid compat_type={compat_type!r}')
        self.compat_type = compat_type
        self.correct_peroxide = correct_peroxide
        self.check_potcar = check_potcar
        self.check_potcar_hash = check_potcar_hash
        if config_file:
            if os.path.isfile(config_file):
                self.config_file: str | None = config_file
                config = loadfn(self.config_file)
            else:
                raise ValueError(f'Custom MaterialsProject2020Compatibility config_file={config_file!r} does not exist.')
        else:
            self.config_file = None
            config = MP2020_COMPAT_CONFIG
        self.name = config['Name']
        self.comp_correction = config['Corrections'].get('CompositionCorrections', defaultdict(float))
        self.comp_errors = config['Uncertainties'].get('CompositionCorrections', defaultdict(float))
        if self.compat_type == 'Advanced':
            self.u_settings = MPRelaxSet.CONFIG['INCAR']['LDAUU']
            self.u_corrections = config['Corrections'].get('GGAUMixingCorrections', defaultdict(float))
            self.u_errors = config['Uncertainties'].get('GGAUMixingCorrections', defaultdict(float))
        else:
            self.u_settings = {}
            self.u_corrections = {}
            self.u_errors = {}

    def get_adjustments(self, entry: AnyComputedEntry) -> list[EnergyAdjustment]:
        """Get the energy adjustments for a ComputedEntry or ComputedStructureEntry.

        Energy corrections are implemented directly in this method instead of in
        separate AnionCorrection, GasCorrection, or UCorrection classes which
        were used in the legacy correction scheme.

        Args:
            entry: A ComputedEntry or ComputedStructureEntry object.

        Returns:
            list[EnergyAdjustment]: A list of EnergyAdjustment to be applied to the Entry.

        Raises:
            CompatibilityError if the entry is not compatible
        """
        if entry.parameters.get('run_type') not in ('GGA', 'GGA+U'):
            raise CompatibilityError(f'Entry {entry.entry_id} has invalid run type {entry.parameters.get('run_type')}. Must be GGA or GGA+U. Discarding.')
        if entry.parameters.get('software', 'vasp') == 'vasp':
            pc = PotcarCorrection(MPRelaxSet, check_hash=self.check_potcar_hash, check_potcar=self.check_potcar)
            pc.get_correction(entry)
        adjustments: list[CompositionEnergyAdjustment] = []
        comp = entry.composition
        rform = comp.reduced_formula
        elements = sorted((el for el in comp.elements if comp[el] > 0), key=lambda el: el.X)
        if len(comp) == 1:
            return adjustments
        if Element('S') in comp:
            sf_type = 'sulfide'
            if entry.data.get('sulfide_type'):
                sf_type = entry.data['sulfide_type']
            elif hasattr(entry, 'structure'):
                sf_type = sulfide_type(entry.structure)
            if sf_type == 'polysulfide':
                sf_type = 'sulfide'
            if sf_type == 'sulfide':
                adjustments.append(CompositionEnergyAdjustment(self.comp_correction['S'], comp['S'], uncertainty_per_atom=self.comp_errors['S'], name='MP2020 anion correction (S)', cls=self.as_dict()))
        if Element('O') in comp:
            if self.correct_peroxide:
                if entry.data.get('oxide_type'):
                    ox_type = entry.data['oxide_type']
                elif hasattr(entry, 'structure'):
                    ox_type = oxide_type(entry.structure, 1.05)
                else:
                    warnings.warn('No structure or oxide_type parameter present. Note that peroxide/superoxide corrections are not as reliable and relies only on detection of special formulas, e.g., Li2O2.')
                    common_peroxides = 'Li2O2 Na2O2 K2O2 Cs2O2 Rb2O2 BeO2 MgO2 CaO2 SrO2 BaO2'.split()
                    common_superoxides = 'LiO2 NaO2 KO2 RbO2 CsO2'.split()
                    ozonides = 'LiO3 NaO3 KO3 NaO5'.split()
                    if rform in common_peroxides:
                        ox_type = 'peroxide'
                    elif rform in common_superoxides:
                        ox_type = 'superoxide'
                    elif rform in ozonides:
                        ox_type = 'ozonide'
                    else:
                        ox_type = 'oxide'
            else:
                ox_type = 'oxide'
            if ox_type == 'hydroxide':
                ox_type = 'oxide'
            adjustments.append(CompositionEnergyAdjustment(self.comp_correction[ox_type], comp['O'], uncertainty_per_atom=self.comp_errors[ox_type], name=f'MP2020 anion correction ({ox_type})', cls=self.as_dict()))
        if 'oxidation_states' not in entry.data:
            try:
                oxi_states = entry.composition.oxi_state_guesses(max_sites=-20)
            except ValueError:
                oxi_states = ({},)
            entry.data['oxidation_states'] = (oxi_states or ({},))[0]
        if entry.data['oxidation_states'] == {}:
            warnings.warn(f'Failed to guess oxidation states for Entry {entry.entry_id} ({entry.reduced_formula}). Assigning anion correction to only the most electronegative atom.')
        for anion in 'Br I Se Si Sb Te H N F Cl'.split():
            if Element(anion) in comp and anion in self.comp_correction:
                apply_correction = False
                if entry.data['oxidation_states'].get(anion, 0) < 0:
                    apply_correction = True
                else:
                    most_electroneg = elements[-1].symbol
                    if anion == most_electroneg:
                        apply_correction = True
                if apply_correction:
                    adjustments.append(CompositionEnergyAdjustment(self.comp_correction[anion], comp[anion], uncertainty_per_atom=self.comp_errors[anion], name=f'MP2020 anion correction ({anion})', cls=self.as_dict()))
        calc_u = entry.parameters.get('hubbards')
        calc_u = defaultdict(int) if calc_u is None else calc_u
        most_electroneg = elements[-1].symbol
        u_corrections = self.u_corrections.get(most_electroneg, defaultdict(float))
        u_settings = self.u_settings.get(most_electroneg, defaultdict(float))
        u_errors = self.u_errors.get(most_electroneg, defaultdict(float))
        for el in comp.elements:
            symbol = el.symbol
            expected_u = float(u_settings.get(symbol, 0))
            actual_u = float(calc_u.get(symbol, 0))
            if actual_u != expected_u:
                raise CompatibilityError(f'Invalid U value of {actual_u:.3} on {symbol}, expected {expected_u:.3}')
            if symbol in u_corrections:
                adjustments.append(CompositionEnergyAdjustment(u_corrections[symbol], comp[el], uncertainty_per_atom=u_errors[symbol], name=f'MP2020 GGA/GGA+U mixing correction ({symbol})', cls=self.as_dict()))
        return adjustments