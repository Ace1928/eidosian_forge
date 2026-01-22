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
def add_abiobjects(self, *abi_objects):
    """
        This function receive a list of AbiVarable objects and add
        the corresponding variables to the input.
        """
    dct = {}
    for obj in abi_objects:
        if not hasattr(obj, 'to_abivars'):
            raise TypeError(f'type {type(obj).__name__} does not have `to_abivars` method')
        dct.update(self.set_vars(obj.to_abivars()))
    return dct