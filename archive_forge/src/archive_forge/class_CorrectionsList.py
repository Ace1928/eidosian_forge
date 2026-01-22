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
class CorrectionsList(Compatibility):
    """The CorrectionsList class combines a list of corrections to be applied to
    an entry or a set of entries. Note that some of the Corrections have
    interdependencies. For example, PotcarCorrection must always be used
    before any other compatibility. Also, AnionCorrection("MP") must be used
    with PotcarCorrection("MP") (similarly with "MIT"). Typically,
    you should use the specific MaterialsProjectCompatibility and
    MITCompatibility subclasses instead.
    """

    def __init__(self, corrections: Sequence[Correction], run_types: list[str] | None=None):
        """
        Args:
            corrections (list[Correction]): Correction objects to apply.
            run_types: Valid DFT run_types for this correction scheme. Entries with run_type
                other than those in this list will be excluded from the list returned
                by process_entries. The default value captures both GGA and GGA+U run types
                historically used by the Materials Project, for example in.
        """
        if run_types is None:
            run_types = ['GGA', 'GGA+U', 'PBE', 'PBE+U']
        self.corrections = corrections
        self.run_types = run_types
        super().__init__()

    def get_adjustments(self, entry: AnyComputedEntry) -> list[EnergyAdjustment]:
        """Get the list of energy adjustments to be applied to an entry."""
        adjustment_list = []
        if entry.parameters.get('run_type') not in self.run_types:
            raise CompatibilityError(f'Entry {entry.entry_id} has invalid run type {entry.parameters.get('run_type')}. Must be GGA or GGA+U. Discarding.')
        corrections, uncertainties = self.get_corrections_dict(entry)
        for k, v in corrections.items():
            uncertainty = np.nan if v != 0 and uncertainties[k] == 0 else uncertainties[k]
            adjustment_list.append(ConstantEnergyAdjustment(v, uncertainty=uncertainty, name=k, cls=self.as_dict()))
        return adjustment_list

    def get_corrections_dict(self, entry: AnyComputedEntry) -> tuple[dict[str, float], dict[str, float]]:
        """Returns the correction values and uncertainties applied to a particular entry.

        Args:
            entry: A ComputedEntry object.

        Returns:
            tuple[dict[str, float], dict[str, float]]: Map from correction names to values
                (1st) and uncertainties (2nd).
        """
        corrections = {}
        uncertainties = {}
        for c in self.corrections:
            val = c.get_correction(entry)
            if val != 0:
                corrections[str(c)] = val.nominal_value
                uncertainties[str(c)] = val.std_dev
        return (corrections, uncertainties)

    def get_explanation_dict(self, entry):
        """Provides an explanation dict of the corrections that are being applied
        for a given compatibility scheme. Inspired by the "explain" methods
        in many database methodologies.

        Args:
            entry: A ComputedEntry.

        Returns:
            dict[str, str | float | list[dict[str, Union[str, float]]]: of the form
                {"Compatibility": "string",
                "Uncorrected_energy": float,
                "Corrected_energy": float,
                "correction_uncertainty:" float,
                "Corrections": [{"Name of Correction": {
                "Value": float, "Explanation": "string", "Uncertainty": float}]}
        """
        corr_entry = self.process_entry(entry)
        uncorrected_energy = (corr_entry or entry).uncorrected_energy
        corrected_energy = corr_entry.energy if corr_entry else None
        correction_uncertainty = corr_entry.correction_uncertainty if corr_entry else None
        dct = {'compatibility': type(self).__name__, 'uncorrected_energy': uncorrected_energy, 'corrected_energy': corrected_energy, 'correction_uncertainty': correction_uncertainty}
        corrections = []
        corr_dict, uncer_dict = self.get_corrections_dict(entry)
        for c in self.corrections:
            if corr_dict.get(str(c), 0) != 0 and uncer_dict.get(str(c), 0) == 0:
                uncer = np.nan
            else:
                uncer = uncer_dict.get(str(c), 0)
            cd = {'name': str(c), 'description': c.__doc__.split('Args')[0].strip(), 'value': corr_dict.get(str(c), 0), 'uncertainty': uncer}
            corrections.append(cd)
        dct['corrections'] = corrections
        return dct

    def explain(self, entry):
        """Prints an explanation of the corrections that are being applied for a
        given compatibility scheme. Inspired by the "explain" methods in many
        database methodologies.

        Args:
            entry: A ComputedEntry.
        """
        dct = self.get_explanation_dict(entry)
        print(f'The uncorrected value of the energy of {entry.composition} is {dct['uncorrected_energy']:f} eV')
        print(f'The following corrections / screening are applied for {dct['compatibility']}:\n')
        for corr in dct['corrections']:
            print(f'{corr['name']} correction: {corr['description']}\n')
            print(f'For the entry, this correction has the value {corr['value']:f} eV.')
            if corr['uncertainty'] != 0 or corr['value'] == 0:
                print(f'This correction has an uncertainty value of {corr['uncertainty']:f} eV.')
            else:
                print('This correction does not have uncertainty data available')
            print('-' * 30)
        print(f'The final energy after corrections is {dct['corrected_energy']:f}')