from __future__ import annotations
import collections
import os
import re
import string
import warnings
from functools import total_ordering
from itertools import combinations_with_replacement, product
from math import isnan
from typing import TYPE_CHECKING, cast
from monty.fractions import gcd, gcd_float
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.core.periodic_table import DummySpecies, Element, ElementType, Species, get_el_sp
from pymatgen.core.units import Mass
from pymatgen.util.string import Stringify, formula_double_format
class ChemicalPotential(dict, MSONable):
    """Class to represent set of chemical potentials. Can be: multiplied/divided by a Number
    multiplied by a Composition (returns an energy) added/subtracted with other ChemicalPotentials.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Args:
            *args: any valid dict init arguments
            **kwargs: any valid dict init arguments.
        """
        dct = dict(*args, **kwargs)
        super().__init__(((get_el_sp(key), val) for key, val in dct.items()))
        if len(dct) != len(self):
            raise ValueError('Duplicate potential specified')

    def __mul__(self, other: object) -> ChemicalPotential:
        if isinstance(other, (int, float)):
            return ChemicalPotential({key: val * other for key, val in self.items()})
        return NotImplemented
    __rmul__ = __mul__

    def __truediv__(self, other: object) -> ChemicalPotential:
        if isinstance(other, (int, float)):
            return ChemicalPotential({key: val / other for key, val in self.items()})
        return NotImplemented
    __div__ = __truediv__

    def __sub__(self, other: object) -> ChemicalPotential:
        if isinstance(other, ChemicalPotential):
            els = {*self} | {*other}
            return ChemicalPotential({e: self.get(e, 0) - other.get(e, 0) for e in els})
        return NotImplemented

    def __add__(self, other: object) -> ChemicalPotential:
        if isinstance(other, ChemicalPotential):
            els = {*self} | {*other}
            return ChemicalPotential({e: self.get(e, 0) + other.get(e, 0) for e in els})
        return NotImplemented

    def get_energy(self, composition: Composition, strict: bool=True) -> float:
        """Calculates the energy of a composition.

        Args:
            composition (Composition): input composition
            strict (bool): Whether all potentials must be specified
        """
        if strict and (missing := (set(composition) - set(self))):
            raise ValueError(f'Potentials not specified for {missing}')
        return sum((self.get(key, 0) * val for key, val in composition.items()))

    def __repr__(self) -> str:
        return f'ChemPots: {super()!r}'