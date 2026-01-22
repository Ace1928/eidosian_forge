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
def _read_fock_matrix(self):
    """Parses the Fock matrix. The matrix is read in whole
        from the output file and then transformed into the right dimensions.
        """
    header_pattern = 'Final Alpha Fock Matrix'
    elements_pattern = '\\-*\\d+\\.\\d+'
    if not self.data.get('unrestricted', []):
        spin_unrestricted = False
        footer_pattern = 'SCF time:'
    else:
        spin_unrestricted = True
        footer_pattern = 'Final Beta Fock Matrix'
    alpha_fock_matrix = read_matrix_pattern(header_pattern, footer_pattern, elements_pattern, self.text, postprocess=float)
    if spin_unrestricted:
        header_pattern = 'Final Beta Fock Matrix'
        footer_pattern = 'SCF time:'
        beta_fock_matrix = read_matrix_pattern(header_pattern, footer_pattern, elements_pattern, self.text, postprocess=float)
    alpha_fock_matrix = process_parsed_fock_matrix(alpha_fock_matrix)
    self.data['alpha_fock_matrix'] = alpha_fock_matrix
    if spin_unrestricted:
        beta_fock_matrix = process_parsed_fock_matrix(beta_fock_matrix)
        self.data['beta_fock_matrix'] = beta_fock_matrix