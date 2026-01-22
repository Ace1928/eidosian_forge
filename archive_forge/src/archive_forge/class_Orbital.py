from __future__ import annotations
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
@unique
class Orbital(Enum):
    """Enum type for specific orbitals. The indices are the order in
    which the orbitals are reported in VASP and has no special meaning.
    """
    s = 0
    py = 1
    pz = 2
    px = 3
    dxy = 4
    dyz = 5
    dz2 = 6
    dxz = 7
    dx2 = 8
    f_3 = 9
    f_2 = 10
    f_1 = 11
    f0 = 12
    f1 = 13
    f2 = 14
    f3 = 15

    def __int__(self) -> int:
        return self.value

    def __str__(self) -> str:
        return str(self.name)

    @property
    def orbital_type(self):
        """Returns OrbitalType of an orbital."""
        return OrbitalType[self.name[0]]