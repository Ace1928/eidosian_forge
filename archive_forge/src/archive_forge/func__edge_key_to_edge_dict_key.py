from __future__ import annotations
import itertools
import logging
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from monty.json import MSONable, jsanitize
from networkx.algorithms.components import is_connected
from networkx.algorithms.traversal import bfs_tree
from pymatgen.analysis.chemenv.connectivity.environment_nodes import EnvironmentNode
from pymatgen.analysis.chemenv.utils.chemenv_errors import ChemenvError
from pymatgen.analysis.chemenv.utils.graph_utils import get_delta
from pymatgen.analysis.chemenv.utils.math_utils import get_linearly_independent_vectors
@staticmethod
def _edge_key_to_edge_dict_key(key):
    if isinstance(key, int):
        return str(key)
    if isinstance(key, str):
        try:
            int(key)
            raise RuntimeError('Cannot pass an edge key which is a str representation of an int.')
        except ValueError:
            return key
    raise ValueError('Edge key should be either a str or an int.')