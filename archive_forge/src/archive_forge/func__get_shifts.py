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
def _get_shifts(shift_mode, structure):
    """
    Gives the shifts based on the selected shift mode and on the symmetry of the structure.
    G: Gamma centered
    M: Monkhorst-Pack ((0.5, 0.5, 0.5))
    S: Symmetric. Respects the chksymbreak with multiple shifts
    O: OneSymmetric. Respects the chksymbreak with a single shift (as in 'S' if a single shift is given, gamma
        centered otherwise.

    Note: for some cases (e.g. body centered tetragonal), both the Symmetric and OneSymmetric may fail to satisfy the
        chksymbreak condition (Abinit input variable).
    """
    if shift_mode == ShiftMode.GammaCentered:
        return ((0, 0, 0),)
    if shift_mode == ShiftMode.MonkhorstPack:
        return ((0.5, 0.5, 0.5),)
    if shift_mode == ShiftMode.Symmetric:
        return calc_shiftk(structure)
    if shift_mode == ShiftMode.OneSymmetric:
        shifts = calc_shiftk(structure)
        if len(shifts) == 1:
            return shifts
        return ((0, 0, 0),)
    raise ValueError(f'Invalid shift_mode={shift_mode!r}')