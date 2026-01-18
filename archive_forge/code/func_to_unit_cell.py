from __future__ import annotations
import collections
import json
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MontyEncoder, MSONable
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import DummySpecies, Element, Species, get_el_sp
from pymatgen.util.coord import pbc_diff
def to_unit_cell(self, in_place=False) -> PeriodicSite | None:
    """Move frac coords to within the unit cell."""
    frac_coords = [np.mod(f, 1) if p else f for p, f in zip(self.lattice.pbc, self.frac_coords)]
    if in_place:
        self.frac_coords = np.array(frac_coords)
        return None
    return PeriodicSite(self.species, frac_coords, self.lattice, properties=self.properties, label=self.label)