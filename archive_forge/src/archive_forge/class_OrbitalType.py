from __future__ import annotations
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
@unique
class OrbitalType(Enum):
    """Enum type for orbital type. Indices are the azimuthal quantum number l."""
    s = 0
    p = 1
    d = 2
    f = 3

    def __str__(self) -> str:
        return str(self.name)