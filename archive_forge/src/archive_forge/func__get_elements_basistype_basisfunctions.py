from __future__ import annotations
import collections
import fnmatch
import os
import re
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import LobsterBandStructureSymmLine
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import Dos, LobsterCompleteDos
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Vasprun, VolumetricData
from pymatgen.util.due import Doi, due
@staticmethod
def _get_elements_basistype_basisfunctions(data):
    begin = False
    end = False
    elements = []
    basistype = []
    basisfunctions = []
    for row in data:
        if begin and (not end):
            splitrow = row.split()
            if splitrow[0] not in ['INFO:', 'WARNING:', 'setting', 'calculating', 'post-processing', 'saving', 'spillings', 'writing']:
                elements += [splitrow[0]]
                basistype += [splitrow[1].replace('(', '').replace(')', '')]
                basisfunctions += [splitrow[2:]]
            else:
                end = True
        if 'setting up local basis functions...' in row:
            begin = True
    return (elements, basistype, basisfunctions)