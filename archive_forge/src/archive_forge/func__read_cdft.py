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
def _read_cdft(self):
    """Parses output from charge- or spin-constrained DFT (CDFT) calculations."""
    temp_dict = read_pattern(self.text, {'constraint': 'Constraint\\s+(\\d+)\\s+:\\s+([\\-\\.0-9]+)', 'multiplier': '\\s*Lam\\s+([\\.\\-0-9]+)'})
    self.data['cdft_constraints_multipliers'] = []
    for const, multip in zip(temp_dict.get('constraint', []), temp_dict.get('multiplier', [])):
        entry = {'index': int(const[0]), 'constraint': float(const[1]), 'multiplier': float(multip[0])}
        self.data['cdft_constraints_multipliers'].append(entry)
    header_pattern = '\\s*CDFT Becke Populations\\s*\\n\\-+\\s*\\n\\s*Atom\\s+Excess Electrons\\s+Population \\(a\\.u\\.\\)\\s+Net Spin'
    table_pattern = '\\s*(?:[0-9]+)\\s+(?:[A-Za-z0-9]+)\\s+([\\-\\.0-9]+)\\s+([\\.0-9]+)\\s+([\\-\\.0-9]+)'
    footer_pattern = '\\s*\\-+'
    becke_table = read_table_pattern(self.text, header_pattern, table_pattern, footer_pattern)
    if becke_table is None or len(becke_table) == 0:
        self.data['cdft_becke_excess_electrons'] = self.data['cdft_becke_population'] = self.data['cdft_becke_net_spin'] = None
    else:
        self.data['cdft_becke_excess_electrons'] = []
        self.data['cdft_becke_population'] = []
        self.data['cdft_becke_net_spin'] = []
        for table in becke_table:
            excess = []
            population = []
            spin = []
            for row in table:
                excess.append(float(row[0]))
                population.append(float(row[1]))
                spin.append(float(row[2]))
            self.data['cdft_becke_excess_electrons'].append(excess)
            self.data['cdft_becke_population'].append(population)
            self.data['cdft_becke_net_spin'].append(spin)