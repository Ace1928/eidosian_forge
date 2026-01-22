from __future__ import annotations
import datetime
import itertools
import logging
import math
import os
import re
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from glob import glob
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.io import reverse_readfile, zopen
from monty.json import MSONable, jsanitize
from monty.os.path import zpath
from monty.re import regrep
from numpy.testing import assert_allclose
from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.core.units import unitized
from pymatgen.electronic_structure.bandstructure import (
from pymatgen.electronic_structure.core import Magmom, Orbital, OrbitalType, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.common import VolumetricData as BaseVolumetricData
from pymatgen.io.core import ParseError
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar
from pymatgen.io.wannier90 import Unk
from pymatgen.util.io_utils import clean_lines, micro_pyawk
from pymatgen.util.num import make_symmetric_matrix_from_upper_tri
def _parse_chemical_shielding_calculation(self, elem):
    calculation = []
    istep = {}
    try:
        struct = self._parse_structure(elem.find('structure'))
    except AttributeError:
        struct = None
    for va in elem.findall('varray'):
        istep[va.attrib['name']] = _parse_vasp_array(va)
    istep['structure'] = struct
    istep['electronic_steps'] = []
    calculation.append(istep)
    for scstep in elem.findall('scstep'):
        try:
            e_steps_dict = {i.attrib['name']: _vasprun_float(i.text) for i in scstep.find('energy').findall('i')}
            cur_ene = e_steps_dict['e_fr_energy']
            min_steps = 1 if len(calculation) >= 1 else self.parameters.get('NELMIN', 5)
            if len(calculation[-1]['electronic_steps']) <= min_steps:
                calculation[-1]['electronic_steps'].append(e_steps_dict)
            else:
                last_ene = calculation[-1]['electronic_steps'][-1]['e_fr_energy']
                if abs(cur_ene - last_ene) < 1.0:
                    calculation[-1]['electronic_steps'].append(e_steps_dict)
                else:
                    calculation.append({'electronic_steps': [e_steps_dict]})
        except AttributeError:
            pass
    calculation[-1].update(calculation[-1]['electronic_steps'][-1])
    return calculation