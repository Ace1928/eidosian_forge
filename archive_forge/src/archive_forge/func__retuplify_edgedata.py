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
def _retuplify_edgedata(edata):
    """
        Private method used to cast back lists to tuples where applicable in an edge data.

        The format of the edge data is:
        {
            "start": STARTINDEX,
            "end": ENDINDEX,
            "delta": TUPLE(DELTAX, DELTAY, DELTAZ),
            "ligands": [
                TUPLE(
                    LIGAND_1_INDEX,
                    TUPLE(DELTAX_START_LIG_1, DELTAY_START_LIG_1, DELTAZ_START_LIG_1),
                    TUPLE(DELTAX_END_LIG_1, DELTAY_END_LIG_1, DELTAZ_END_LIG_1),
                ),
                TUPLE(LIGAND_2_INDEX, ...),
                ...,
            ],
        }
        When serializing to json/bson, these tuples are transformed into lists. This method transforms these lists
        back to tuples.

        Args:
            edata (dict): Edge data dictionary with possibly the above tuples as lists.

        Returns:
            dict: Edge data dictionary with the lists transformed back into tuples when applicable.
        """
    edata['delta'] = tuple(edata['delta'])
    edata['ligands'] = [(lig[0], tuple(lig[1]), tuple(lig[2])) for lig in edata['ligands']]
    return edata