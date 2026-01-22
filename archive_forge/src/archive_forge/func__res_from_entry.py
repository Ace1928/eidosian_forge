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
def _res_from_entry(cls, entry: ComputedStructureEntry) -> Res:
    """Produce a res file structure from a pymatgen ComputedStructureEntry."""
    seed = entry.data.get('seed') or str(hash(entry))
    pres = float(entry.data.get('pressure', 0))
    isd = float(entry.data.get('isd', 0))
    iasd = float(entry.data.get('iasd', 0))
    spg, _ = entry.structure.get_space_group_info()
    rems = [str(x) for x in entry.data.get('rems', [])]
    return Res(AirssTITL(seed, pres, entry.structure.volume, entry.energy, isd, iasd, spg, 1), rems, cls._cell_from_lattice(entry.structure.lattice), cls._sfac_from_sites(list(entry.structure)))