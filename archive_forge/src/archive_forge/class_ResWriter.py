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
class ResWriter:
    """This class provides a means to write a Structure or ComputedStructureEntry to a res file."""

    @classmethod
    def _cell_from_lattice(cls, lattice: Lattice) -> ResCELL:
        """Produce CELL entry from a pymatgen Lattice."""
        return ResCELL(1.0, lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma)

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

    @classmethod
    def _sfac_from_sites(cls, sites: list[PeriodicSite]) -> ResSFAC:
        """Produce a SFAC block from a list of pymatgen PeriodicSite."""
        ions = cls._ions_from_sites(sites)
        species = {ion.specie for ion in ions}
        return ResSFAC(species, ions)

    @classmethod
    def _res_from_structure(cls, structure: Structure) -> Res:
        """Produce a res file structure from a pymatgen Structure."""
        return Res(None, [], cls._cell_from_lattice(structure.lattice), cls._sfac_from_sites(list(structure)))

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

    def __init__(self, entry: Structure | ComputedStructureEntry):
        """This class can be constructed from either a pymatgen Structure or ComputedStructureEntry object."""
        func: Callable[[Structure], Res] | Callable[[ComputedStructureEntry], Res]
        func = self._res_from_structure
        if isinstance(entry, ComputedStructureEntry):
            func = self._res_from_entry
        self._res = func(entry)

    def __str__(self):
        return str(self._res)

    @property
    def string(self) -> str:
        """The contents of the res file."""
        return str(self)

    def write(self, filename: str) -> None:
        """Write the res data to a file."""
        with zopen(filename, mode='w') as file:
            file.write(str(self))