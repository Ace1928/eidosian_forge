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
class ManualEnergyAdjustment(ConstantEnergyAdjustment):
    """A manual energy adjustment applied to a ComputedEntry."""

    def __init__(self, value):
        """
        Args:
            value: float, value of the energy adjustment in eV.
        """
        name = 'Manual energy adjustment'
        description = 'Manual energy adjustment'
        super().__init__(value, name=name, cls=None, description=description)