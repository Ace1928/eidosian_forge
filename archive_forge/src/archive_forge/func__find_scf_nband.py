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
def _find_scf_nband(structure, pseudos, electrons, spinat=None):
    """Find the value of nband."""
    if electrons.nband is not None:
        return electrons.nband
    nsppol, smearing = (electrons.nsppol, electrons.smearing)
    n_val_elec = num_valence_electrons(structure, pseudos)
    n_val_elec -= electrons.charge
    nband = n_val_elec // 2
    nband = max(np.ceil(nband * 1.2), nband + 10) if smearing else max(np.ceil(nband * 1.1), nband + 4)
    if nsppol == 2 and spinat is not None:
        nband += np.ceil(max(np.sum(spinat, axis=0)) / 2.0)
    nband += nband % 2
    return int(nband)