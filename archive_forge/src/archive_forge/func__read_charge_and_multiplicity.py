from __future__ import annotations
import copy
import logging
import math
import os
import re
import struct
import warnings
from typing import TYPE_CHECKING, Any
import networkx as nx
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core import Molecule
from pymatgen.io.qchem.utils import (
def _read_charge_and_multiplicity(self):
    """Parses charge and multiplicity."""
    temp_charge = read_pattern(self.text, {'key': '\\$molecule\\s+([\\-\\d]+)\\s+\\d'}, terminate_on_match=True).get('key')
    if temp_charge is not None:
        self.data['charge'] = int(temp_charge[0][0])
    else:
        temp_charge = read_pattern(self.text, {'key': 'Sum of atomic charges \\=\\s+([\\d\\-\\.\\+]+)'}, terminate_on_match=True).get('key')
        if temp_charge is None:
            self.data['charge'] = None
        else:
            self.data['charge'] = int(float(temp_charge[0][0]))
    temp_multiplicity = read_pattern(self.text, {'key': '\\$molecule\\s+[\\-\\d]+\\s+(\\d)'}, terminate_on_match=True).get('key')
    if temp_multiplicity is not None:
        self.data['multiplicity'] = int(temp_multiplicity[0][0])
    else:
        temp_multiplicity = read_pattern(self.text, {'key': 'Sum of spin\\s+charges \\=\\s+([\\d\\-\\.\\+]+)'}, terminate_on_match=True).get('key')
        if temp_multiplicity is None:
            self.data['multiplicity'] = 1
        else:
            self.data['multiplicity'] = int(float(temp_multiplicity[0][0])) + 1