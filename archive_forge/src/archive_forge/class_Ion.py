from and back to a string/file is not guaranteed to be reversible, i.e. a diff on the output
from __future__ import annotations
import datetime
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element, Lattice, PeriodicSite, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.core import ParseError
@dataclass(frozen=True)
class Ion:
    specie: str
    specie_num: int
    pos: Vector3D
    occupancy: float
    spin: float | None

    def __str__(self) -> str:
        if self.spin is None:
            ion_fmt = '{:<7s}{:<2d} {:.8f} {:.8f} {:.8f} {:f}'
            return ion_fmt.format(self.specie, self.specie_num, *self.pos, self.occupancy)
        ion_fmt = '{:<7s}{:<2d} {:.8f} {:.8f} {:.8f} {:f} {:5.2f}'
        return ion_fmt.format(self.specie, self.specie_num, *self.pos, self.occupancy, self.spin)