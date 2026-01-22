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
def _read_nbo_data(self):
    """Parses NBO output."""
    dfs = nbo_parser(self.filename)
    nbo_data = {}
    for key, value in dfs.items():
        nbo_data[key] = [df.to_dict() for df in value]
    self.data['nbo_data'] = nbo_data