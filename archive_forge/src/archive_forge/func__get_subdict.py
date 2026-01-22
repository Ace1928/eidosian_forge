from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Kpoints
def _get_subdict(master_dict, subkeys):
    """Helper method to get a set of keys from a larger dictionary."""
    return {k: master_dict[k] for k in subkeys if k in master_dict and master_dict[k] is not None}