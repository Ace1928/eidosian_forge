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
class ConstantEnergyAdjustment(EnergyAdjustment):
    """A constant energy adjustment applied to a ComputedEntry. Useful in energy referencing
    schemes such as the Aqueous energy referencing scheme.
    """

    def __init__(self, value, uncertainty=np.nan, name='Constant energy adjustment', cls=None, description='Constant energy adjustment'):
        """
        Args:
            value: float, value of the energy adjustment in eV
            uncertainty: float, uncertainty of the energy adjustment in eV. (Default: np.nan)
            name: str, human-readable name of the energy adjustment.
                (Default: Constant energy adjustment)
            cls: dict, Serialized Compatibility class used to generate the energy
                adjustment. (Default: None)
            description: str, human-readable explanation of the energy adjustment.
        """
        super().__init__(value, uncertainty, name=name, cls=cls, description=description)
        self._value = value
        self._uncertainty = uncertainty

    @property
    def explain(self):
        """Return an explanation of how the energy adjustment is calculated."""
        return f'{self.description} ({self.value:.3f} eV)'

    def normalize(self, factor: float) -> None:
        """Normalize energy adjustment (in place), dividing value/uncertainty by a
        factor.

        Args:
            factor: factor to divide by.
        """
        self._value /= factor
        self._uncertainty /= factor