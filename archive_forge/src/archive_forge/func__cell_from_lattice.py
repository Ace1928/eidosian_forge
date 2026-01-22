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
@classmethod
def _cell_from_lattice(cls, lattice: Lattice) -> ResCELL:
    """Produce CELL entry from a pymatgen Lattice."""
    return ResCELL(1.0, lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma)