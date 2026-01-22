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
def _parse_cell(self, line: str) -> ResCELL:
    """Parses the CELL entry."""
    fields = line.split()
    if len(fields) != 7:
        raise ResParseError(f'Failed to parse CELL line={line!r}, expected 7 fields.')
    field_1, a, b, c, alpha, beta, gamma = map(float, fields)
    return ResCELL(field_1, a, b, c, alpha, beta, gamma)