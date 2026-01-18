from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import Kpoint
def write_phononwebsite(self, filename: str | PathLike) -> None:
    """Write a json file for the phononwebsite:
        http://henriquemiranda.github.io/phononwebsite.
        """
    with open(filename, mode='w') as file:
        json.dump(self.as_phononwebsite(), file)