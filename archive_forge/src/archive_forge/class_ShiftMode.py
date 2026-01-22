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
@unique
class ShiftMode(Enum):
    """
    Class defining the mode to be used for the shifts.
    G: Gamma centered
    M: Monkhorst-Pack ((0.5, 0.5, 0.5))
    S: Symmetric. Respects the chksymbreak with multiple shifts
    O: OneSymmetric. Respects the chksymbreak with a single shift (as in 'S' if a single shift is given, gamma
        centered otherwise.
    """
    GammaCentered = 'G'
    MonkhorstPack = 'M'
    Symmetric = 'S'
    OneSymmetric = 'O'

    @classmethod
    def from_object(cls, obj) -> Self:
        """
        Returns an instance of ShiftMode based on the type of object passed. Converts strings to ShiftMode depending
        on the initial letter of the string. G for GammaCentered, M for MonkhorstPack,
        S for Symmetric, O for OneSymmetric.
        Case insensitive.
        """
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, str):
            return cls(obj[0].upper())
        raise TypeError(f'The object provided is not handled: type {type(obj).__name__}')