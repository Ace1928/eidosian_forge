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
def _remove_node_deps(self, jobid, crashfile, graph):
    import networkx as nx
    try:
        dfs_preorder = nx.dfs_preorder
    except AttributeError:
        dfs_preorder = nx.dfs_preorder_nodes
    subnodes = [s for s in dfs_preorder(graph, self.procs[jobid])]
    for node in subnodes:
        idx = self.procs.index(node)
        self.proc_done[idx] = True
        self.proc_pending[idx] = False
    return dict(node=self.procs[jobid], dependents=subnodes, crashfile=crashfile)