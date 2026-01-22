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
class ResSFAC:
    species: set[str]
    ions: list[Ion]

    def __str__(self) -> str:
        species = ' '.join((f'{specie:<2s}' for specie in self.species))
        ions = '\n'.join(map(str, self.ions))
        return f'SFAC {species}\n{ions}\nEND\n'