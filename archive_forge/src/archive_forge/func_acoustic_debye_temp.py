from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from monty.dev import requires
from monty.json import MSONable
from scipy.interpolate import UnivariateSpline
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.units import amu_to_kg
from pymatgen.phonon.bandstructure import PhononBandStructure, PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
@property
def acoustic_debye_temp(self) -> float:
    """Acoustic Debye temperature in K, i.e. the Debye temperature divided by n_sites**(1/3).
        Adapted from abipy.
        """
    assert self.structure is not None, 'Structure is not defined.'
    return self.debye_temp_limit / len(self.structure) ** (1 / 3)