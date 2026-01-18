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
def update_charge_from_potcar(self, path):
    """
        Sets the charge of a structure based on the POTCARs found.

        Args:
            path: Path to search for POTCARs
        """
    potcar = self.get_potcars(path)
    if potcar and self.incar.get('ALGO', '') not in ['GW0', 'G0W0', 'GW', 'BSE']:
        nelect = self.parameters['NELECT']
        if len(potcar) == len(self.initial_structure.composition.element_composition):
            potcar_nelect = sum((self.initial_structure.composition.element_composition[ps.element] * ps.ZVAL for ps in potcar))
        else:
            nums = [len(list(g)) for _, g in itertools.groupby(self.atomic_symbols)]
            potcar_nelect = sum((ps.ZVAL * num for ps, num in zip(potcar, nums)))
        charge = potcar_nelect - nelect
        for s in self.structures:
            s._charge = charge
        if hasattr(self, 'initial_structure'):
            self.initial_structure._charge = charge
        if hasattr(self, 'final_structure'):
            self.final_structure._charge = charge