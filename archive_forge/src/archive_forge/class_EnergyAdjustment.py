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
class EnergyAdjustment(MSONable):
    """Lightweight class to contain information about an energy adjustment or
    energy correction.
    """

    def __init__(self, value, uncertainty=np.nan, name='Manual adjustment', cls=None, description=''):
        """
        Args:
            value (float): value of the energy adjustment in eV
            uncertainty (float): uncertainty of the energy adjustment in eV. Default: np.nan
            name (str): human-readable name of the energy adjustment.
                (Default: Manual adjustment)
            cls (dict): Serialized Compatibility class used to generate the energy adjustment. Defaults to {}.
            description (str): human-readable explanation of the energy adjustment.
        """
        self.name = name
        self.cls = cls or {}
        self.description = description
        self._value = value
        self._uncertainty = uncertainty

    @property
    def value(self):
        """Return the value of the energy correction in eV."""
        return self._value

    @property
    def uncertainty(self):
        """Return the uncertainty in the value of the energy adjustment in eV."""
        return self._uncertainty

    @abc.abstractmethod
    def normalize(self, factor):
        """Scale the value of the current energy adjustment by factor in-place.

        This method is utilized in ComputedEntry.normalize() to scale the energies to a formula unit basis
        (e.g. E_Fe6O9 = 3 x E_Fe2O3).
        """

    @property
    @abc.abstractmethod
    def explain(self):
        """Return an explanation of how the energy adjustment is calculated."""

    def __repr__(self):
        name, value, uncertainty, description = (self.name, float(self.value), self.uncertainty, self.description)
        generated_by = self.cls.get('@class', 'unknown') if isinstance(self.cls, dict) else type(self.cls).__name__
        return f'{type(self).__name__}(name={name!r}, value={value:.3}, uncertainty={uncertainty:.3}, description={description!r}, generated_by={generated_by!r})'