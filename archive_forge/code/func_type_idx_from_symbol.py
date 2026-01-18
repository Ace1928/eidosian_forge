from __future__ import annotations
import logging
import os.path
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.collections import AttrDict
from monty.dev import requires
from monty.functools import lazy_property
from monty.string import marquee
from pymatgen.core.structure import Structure
from pymatgen.core.units import ArrayWithUnit
from pymatgen.core.xcfunc import XcFunc
def type_idx_from_symbol(self, symbol):
    """Returns the type index from the chemical symbol. Note python convention."""
    return self.chemical_symbols.index(symbol)