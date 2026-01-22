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
class ComputedStructureEntry(ComputedEntry):
    """A heavier version of ComputedEntry which contains a structure as well. The
    structure is needed for some analyses.
    """

    def __init__(self, structure: Structure, energy: float, correction: float=0.0, composition: Composition | str | dict[str, float] | None=None, energy_adjustments: list | None=None, parameters: dict | None=None, data: dict | None=None, entry_id: object | None=None) -> None:
        """Initializes a ComputedStructureEntry.

        Args:
            structure (Structure): The actual structure of an entry.
            energy (float): Energy of the entry. Usually the final calculated
                energy from VASP or other electronic structure codes.
            correction (float, optional): A correction to the energy. This is mutually exclusive with
                energy_adjustments, i.e. pass either or neither but not both. Defaults to 0.
            composition (Composition): Composition of the entry. For
                flexibility, this can take the form of all the typical input
                taken by a Composition, including a {symbol: amt} dict,
                a string formula, and others.
            energy_adjustments: An optional list of EnergyAdjustment to
                be applied to the energy. This is used to modify the energy for
                certain analyses. Defaults to None.
            parameters: An optional dict of parameters associated with
                the entry. Defaults to None.
            data: An optional dict of any additional data associated
                with the entry. Defaults to None.
            entry_id: An optional id to uniquely identify the entry.
        """
        if composition:
            composition = Composition(composition)
            if composition.get_integer_formula_and_factor()[0] != structure.composition.get_integer_formula_and_factor()[0]:
                raise ValueError('Mismatching composition provided.')
        else:
            composition = structure.composition
        super().__init__(composition, energy, correction=correction, energy_adjustments=energy_adjustments, parameters=parameters, data=data, entry_id=entry_id)
        self._structure = structure

    @property
    def structure(self) -> Structure:
        """The structure of the entry."""
        return self._structure

    def as_dict(self) -> dict:
        """MSONable dict."""
        dct = super().as_dict()
        dct['structure'] = self.structure.as_dict()
        return dct

    @classmethod
    def from_dict(cls, dct) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            ComputedStructureEntry
        """
        if dct['correction'] != 0 and (not dct.get('energy_adjustments')):
            struct = MontyDecoder().process_decoded(dct['structure'])
            return cls(struct, dct['energy'], correction=dct['correction'], parameters={k: MontyDecoder().process_decoded(v) for k, v in dct.get('parameters', {}).items()}, data={k: MontyDecoder().process_decoded(v) for k, v in dct.get('data', {}).items()}, entry_id=dct.get('entry_id'))
        return cls(MontyDecoder().process_decoded(dct['structure']), dct['energy'], composition=dct.get('composition'), correction=0, energy_adjustments=[MontyDecoder().process_decoded(e) for e in dct.get('energy_adjustments', {})], parameters={k: MontyDecoder().process_decoded(v) for k, v in dct.get('parameters', {}).items()}, data={k: MontyDecoder().process_decoded(v) for k, v in dct.get('data', {}).items()}, entry_id=dct.get('entry_id'))

    def normalize(self, mode: Literal['formula_unit', 'atom']='formula_unit') -> ComputedStructureEntry:
        """Normalize the entry's composition and energy. The structure remains unchanged.

        Args:
            mode ("formula_unit" | "atom"): "formula_unit" (the default) normalizes to composition.reduced_formula.
                "atom" normalizes such that the composition amounts sum to 1.
        """
        warnings.warn(f'Normalization of a `{type(self).__name__}` makes `self.composition` and `self.structure.composition` inconsistent - please use self.composition for all further calculations.')
        factor = self._normalization_factor(mode)
        dct = super().normalize(mode).as_dict()
        dct['structure'] = self.structure.as_dict()
        entry = self.from_dict(dct)
        entry._composition /= factor
        return entry

    def copy(self) -> ComputedStructureEntry:
        """Returns a copy of the ComputedStructureEntry."""
        return ComputedStructureEntry(structure=self.structure.copy(), energy=self.uncorrected_energy, composition=self.composition, energy_adjustments=self.energy_adjustments, parameters=self.parameters, data=self.data, entry_id=self.entry_id)