from __future__ import annotations
import abc
import json
import math
import os
import warnings
from itertools import combinations
from typing import TYPE_CHECKING, Literal, cast
import numpy as np
from monty.json import MontyDecoder, MontyEncoder, MSONable
from scipy.interpolate import interp1d
from uncertainties import ufloat
from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from pymatgen.util.due import Doi, due
class ComputedEntry(Entry):
    """Lightweight Entry object for computed data. Contains facilities
    for applying corrections to the energy attribute and for storing
    calculation parameters.
    """

    def __init__(self, composition: Composition | str | dict[str, float], energy: float, correction: float=0.0, energy_adjustments: list | None=None, parameters: dict | None=None, data: dict | None=None, entry_id: object | None=None):
        """Initializes a ComputedEntry.

        Args:
            composition (Composition): Composition of the entry. For
                flexibility, this can take the form of all the typical input
                taken by a Composition, including a {symbol: amt} dict,
                a string formula, and others.
            energy (float): Energy of the entry. Usually the final calculated
                energy from VASP or other electronic structure codes.
            correction (float): Manually set an energy correction, will ignore
                energy_adjustments if specified.
            energy_adjustments: An optional list of EnergyAdjustment to
                be applied to the energy. This is used to modify the energy for
                certain analyses. Defaults to None.
            parameters: An optional dict of parameters associated with
                the entry. Defaults to None.
            data: An optional dict of any additional data associated
                with the entry. Defaults to None.
            entry_id: An optional id to uniquely identify the entry.
        """
        super().__init__(composition, energy)
        self.energy_adjustments = energy_adjustments or []
        if correction != 0.0:
            if energy_adjustments:
                raise ValueError(f'Argument conflict! Setting correction = {correction:.3f} conflicts with setting energy_adjustments. Specify one or the other.')
            self.correction = correction
        self.parameters = parameters or {}
        self.data = data or {}
        self.entry_id = entry_id
        self.name = self.reduced_formula

    @property
    def uncorrected_energy(self) -> float:
        """
        Returns:
            float: the *uncorrected* energy of the entry.
        """
        return self._energy

    @property
    def energy(self) -> float:
        """The *corrected* energy of the entry."""
        return self.uncorrected_energy + self.correction

    @property
    def uncorrected_energy_per_atom(self) -> float:
        """
        Returns:
            float: the *uncorrected* energy of the entry, normalized by atoms in eV/atom.
        """
        return self.uncorrected_energy / self.composition.num_atoms

    @property
    def correction(self) -> float:
        """
        Returns:
            float: the total energy correction / adjustment applied to the entry in eV.
        """
        corr = ufloat(0.0, 0.0) + sum((ufloat(ea.value, ea.uncertainty) for ea in self.energy_adjustments))
        return corr.nominal_value

    @correction.setter
    def correction(self, x: float) -> None:
        corr = ManualEnergyAdjustment(x)
        self.energy_adjustments = [corr]

    @property
    def correction_per_atom(self) -> float:
        """
        Returns:
            float: the total energy correction / adjustment applied to the entry in eV/atom.
        """
        return self.correction / self.composition.num_atoms

    @property
    def correction_uncertainty(self) -> float:
        """
        Returns:
            float: the uncertainty of the energy adjustments applied to the entry in eV.
        """
        unc = ufloat(0.0, 0.0) + sum((ufloat(ea.value, ea.uncertainty) if not np.isnan(ea.uncertainty) else ufloat(ea.value, 0) for ea in self.energy_adjustments))
        if unc.nominal_value != 0 and unc.std_dev == 0:
            return np.nan
        return unc.std_dev

    @property
    def correction_uncertainty_per_atom(self) -> float:
        """
        Returns:
            float: the uncertainty of the energy adjustments applied to the entry in eV/atom.
        """
        return self.correction_uncertainty / self.composition.num_atoms

    def normalize(self, mode: Literal['formula_unit', 'atom']='formula_unit') -> ComputedEntry:
        """Normalize the entry's composition and energy.

        Args:
            mode ("formula_unit" | "atom"): "formula_unit" (the default) normalizes to composition.reduced_formula.
                "atom" normalizes such that the composition amounts sum to 1.
        """
        factor = self._normalization_factor(mode)
        new_composition = self._composition / factor
        new_energy = self._energy / factor
        new_entry_dict = self.as_dict()
        new_entry_dict['composition'] = new_composition.as_dict()
        new_entry_dict['energy'] = new_energy
        new_energy_adjustments = MontyDecoder().process_decoded(new_entry_dict['energy_adjustments'])
        for ea in new_energy_adjustments:
            ea.normalize(factor)
        new_entry_dict['energy_adjustments'] = [ea.as_dict() for ea in new_energy_adjustments]
        return self.from_dict(new_entry_dict)

    def __repr__(self) -> str:
        n_atoms = self.composition.num_atoms
        output = [f'{self.entry_id} {type(self).__name__:<10} - {self.formula:<12} ({self.reduced_formula})', f'{'Energy (Uncorrected)':<24} = {self._energy:<9.4f} eV ({self._energy / n_atoms:<8.4f} eV/atom)', f'{'Correction':<24} = {self.correction:<9.4f} eV ({self.correction / n_atoms:<8.4f} eV/atom)', f'{'Energy (Final)':<24} = {self.energy:<9.4f} eV ({self.energy_per_atom:<8.4f} eV/atom)', 'Energy Adjustments:']
        if len(self.energy_adjustments) == 0:
            output.append('  None')
        else:
            for e in self.energy_adjustments:
                output.append(f'  {e.name:<23}: {e.value:<9.4f} eV ({e.value / n_atoms:<8.4f} eV/atom)')
        output.append('Parameters:')
        for k, v in self.parameters.items():
            output.append(f'  {k:<22} = {v}')
        output.append('Data:')
        for k, v in self.data.items():
            output.append(f'  {k:<22} = {v}')
        return '\n'.join(output)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        needed_attrs = ('composition', 'energy', 'entry_id')
        if not all((hasattr(other, attr) for attr in needed_attrs)):
            return NotImplemented
        other = cast(ComputedEntry, other)
        if getattr(self, 'entry_id', None) and getattr(other, 'entry_id', None) and (self.entry_id != other.entry_id):
            return False
        if not math.isclose(self.energy, other.energy):
            return False
        if self.composition != other.composition:
            return False
        return True

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
           dct (dict): Dict representation.

        Returns:
            ComputedEntry
        """
        if dct['correction'] != 0 and (not dct.get('energy_adjustments')):
            return cls(dct['composition'], dct['energy'], dct['correction'], parameters={k: MontyDecoder().process_decoded(v) for k, v in dct.get('parameters', {}).items()}, data={k: MontyDecoder().process_decoded(v) for k, v in dct.get('data', {}).items()}, entry_id=dct.get('entry_id'))
        return cls(dct['composition'], dct['energy'], correction=0, energy_adjustments=[MontyDecoder().process_decoded(e) for e in dct.get('energy_adjustments', {})], parameters={k: MontyDecoder().process_decoded(v) for k, v in dct.get('parameters', {}).items()}, data={k: MontyDecoder().process_decoded(v) for k, v in dct.get('data', {}).items()}, entry_id=dct.get('entry_id'))

    def as_dict(self) -> dict:
        """MSONable dict."""
        return_dict = super().as_dict()
        return_dict.update({'entry_id': self.entry_id, 'correction': self.correction, 'energy_adjustments': json.loads(json.dumps(self.energy_adjustments, cls=MontyEncoder)), 'parameters': json.loads(json.dumps(self.parameters, cls=MontyEncoder)), 'data': json.loads(json.dumps(self.data, cls=MontyEncoder))})
        return return_dict

    def __hash__(self) -> int:
        if self.entry_id is not None:
            return hash(f'{type(self).__name__}{self.entry_id}')
        return super().__hash__()

    def copy(self) -> ComputedEntry:
        """Returns a copy of the ComputedEntry."""
        return ComputedEntry(composition=self.composition, energy=self.uncorrected_energy, energy_adjustments=self.energy_adjustments, parameters=self.parameters, data=self.data, entry_id=self.entry_id)