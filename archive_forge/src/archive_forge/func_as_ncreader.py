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
def as_ncreader(file):
    """
    Convert file into a NetcdfReader instance.
    Returns reader, close_it where close_it is set to True
    if we have to close the file before leaving the procedure.
    """
    return _as_reader(file, NetcdfReader)