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
def _read_coefficient_matrix(self):
    """Parses the coefficient matrix from the output file. Done is much
        the same was as the Fock matrix.
        """
    header_pattern = 'Final Alpha MO Coefficients'
    elements_pattern = '\\-*\\d+\\.\\d+'
    if not self.data.get('unrestricted', []):
        spin_unrestricted = False
        footer_pattern = 'Final Alpha Density Matrix'
    else:
        spin_unrestricted = True
        footer_pattern = 'Final Beta MO Coefficients'
    alpha_coeff_matrix = read_matrix_pattern(header_pattern, footer_pattern, elements_pattern, self.text, postprocess=float)
    if spin_unrestricted:
        header_pattern = 'Final Beta MO Coefficients'
        footer_pattern = 'Final Alpha Density Matrix'
        beta_coeff_matrix = read_matrix_pattern(header_pattern, footer_pattern, elements_pattern, self.text, postprocess=float)
    alpha_coeff_matrix = process_parsed_fock_matrix(alpha_coeff_matrix)
    self.data['alpha_coeff_matrix'] = alpha_coeff_matrix
    if spin_unrestricted:
        beta_coeff_matrix = process_parsed_fock_matrix(beta_coeff_matrix)
        self.data['beta_coeff_matrix'] = beta_coeff_matrix