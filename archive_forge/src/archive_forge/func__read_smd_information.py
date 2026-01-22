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
def _read_smd_information(self):
    """Parses information from SMD solvent calculations."""
    temp_dict = read_pattern(self.text, {'smd0': 'E-EN\\(g\\) gas\\-phase elect\\-nuc energy\\s*([\\d\\-\\.]+) a\\.u\\.', 'smd3': 'G\\-ENP\\(liq\\) elect\\-nuc\\-pol free energy of system\\s*([\\d\\-\\.]+) a\\.u\\.', 'smd4': 'G\\-CDS\\(liq\\) cavity\\-dispersion\\-solvent structure\\s*free energy\\s*([\\d\\-\\.]+) kcal\\/mol', 'smd6': 'G\\-S\\(liq\\) free energy of system\\s*([\\d\\-\\.]+) a\\.u\\.', 'smd9': 'DeltaG\\-S\\(liq\\) free energy of\\s*solvation\\s*\\(9\\) = \\(6\\) \\- \\(0\\)\\s*([\\d\\-\\.]+) kcal\\/mol'})
    for key in temp_dict:
        if temp_dict.get(key) is None:
            self.data['solvent_data'][key] = None
        elif len(temp_dict.get(key)) == 1:
            self.data['solvent_data'][key] = float(temp_dict.get(key)[0][0])
        else:
            temp_result = np.zeros(len(temp_dict.get(key)))
            for ii, entry in enumerate(temp_dict.get(key)):
                temp_result[ii] = float(entry[0])
            self.data['solvent_data'][key] = temp_result