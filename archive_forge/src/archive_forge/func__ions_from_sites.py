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
def _ions_from_sites(cls, sites: list[PeriodicSite]) -> list[Ion]:
    """Produce a list of entries for a SFAC block from a list of pymatgen PeriodicSite."""
    ions: list[Ion] = []
    i = 0
    for site in sites:
        for specie, occ in site.species.items():
            i += 1
            x, y, z = map(float, site.frac_coords)
            spin = site.properties.get('magmom')
            spin = spin and float(spin)
            ions.append(Ion(specie, i, (x, y, z), occ, spin))
    return ions