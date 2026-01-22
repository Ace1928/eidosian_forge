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
def _stopping_criterion(run_level, accuracy):
    """Return the stopping criterion for this run_level with the given accuracy."""
    _run_level_tolname_map = {'scf': 'tolvrs', 'nscf': 'tolwfr', 'dfpt': 'toldfe', 'screening': 'toldfe', 'sigma': 'toldfe', 'bse': 'toldfe', 'relax': 'tolrff'}
    tol_name = _run_level_tolname_map[run_level]
    return {tol_name: getattr(_tolerances[tol_name], accuracy)}