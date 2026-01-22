from __future__ import annotations
import abc
import copy
import json
import logging
import os
from collections import namedtuple
from collections.abc import Mapping, MutableMapping, Sequence
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.collections import AttrDict
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.io.abinit import abiobjects as aobj
from pymatgen.io.abinit.pseudos import Pseudo, PseudoTable
from pymatgen.io.abinit.variable import InputVariable
from pymatgen.symmetry.bandstructure import HighSymmKpath
def as_structure(obj):
    """
    Convert obj into a Structure. Accepts:

        - Structure object.
        - Filename
        - Dictionaries (MSONable format or dictionaries with abinit variables).
    """
    if isinstance(obj, Structure):
        return obj
    if isinstance(obj, str):
        return Structure.from_file(obj)
    if isinstance(obj, Mapping):
        if '@module' in obj:
            return Structure.from_dict(obj)
        return aobj.structure_from_abivars(cls=None, **obj)
    raise TypeError(f"Don't know how to convert {type(obj)} into a structure")