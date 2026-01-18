from __future__ import annotations
from math import isclose
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from pymatgen.core.units import kb as kb_ev
from pymatgen.util.due import Doi, due

        Calculate Quasi-RRHO thermochemistry

        Args:
            mol (Molecule): Pymatgen molecule
            mult (int): Spin multiplicity
            sigma_r (int): Rotational symmetry number
            frequencies (list): List of frequencies [cm^-1]
            elec_energy (float): Electronic energy [Ha]
        