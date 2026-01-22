import sys
from copy import deepcopy
from glob import glob
import os
import shutil
from time import sleep, time
from traceback import format_exception
import numpy as np
from ... import logging
from ...utils.misc import str2bool
from ..engine.utils import topological_sort, load_resultfile
from ..engine import MapNode
from .tools import report_crash, report_nodes_not_run, create_pyscript
def _graph_to_lil_matrix(graph, nodelist):
    """Provide a sparse linked list matrix across various NetworkX versions"""
    import scipy.sparse as ssp
    try:
        from networkx import to_scipy_sparse_array
    except ImportError:
        from networkx import to_scipy_sparse_matrix as to_scipy_sparse_array
    return ssp.lil_matrix(to_scipy_sparse_array(graph, nodelist=nodelist, format='lil'))