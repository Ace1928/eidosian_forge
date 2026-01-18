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
Create PeriodicSite from dict representation.

        Args:
            dct (dict): dict representation of PeriodicSite
            lattice: Optional lattice to override lattice specified in d.
                Useful for ensuring all sites in a structure share the same
                lattice.

        Returns:
            PeriodicSite
        