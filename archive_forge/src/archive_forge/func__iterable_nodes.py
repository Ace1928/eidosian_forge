import os
import sys
import pickle
from collections import defaultdict
import re
from copy import deepcopy
from glob import glob
from pathlib import Path
from traceback import format_exception
from hashlib import sha1
from functools import reduce
import numpy as np
from ... import logging, config
from ...utils.filemanip import (
from ...utils.misc import str2bool
from ...utils.functions import create_function_from_source
from ...interfaces.base.traits_extension import (
from ...interfaces.base.support import Bunch, InterfaceResult
from ...interfaces.base import CommandLine
from ...interfaces.utility import IdentityInterface
from ...utils.provenance import ProvStore, pm, nipype_ns, get_id
from inspect import signature
def _iterable_nodes(graph_in):
    """Returns the iterable nodes in the given graph and their join
    dependencies.

    The nodes are ordered as follows:

    - nodes without an itersource precede nodes with an itersource
    - nodes without an itersource are sorted in reverse topological order
    - nodes with an itersource are sorted in topological order

    This order implies the following:

    - every iterable node without an itersource is expanded before any
      node with an itersource

    - every iterable node without an itersource is expanded before any
      of it's predecessor iterable nodes without an itersource

    - every node with an itersource is expanded before any of it's
      successor nodes with an itersource

    Return the iterable nodes list
    """
    import networkx as nx
    nodes = nx.topological_sort(graph_in)
    inodes = [node for node in nodes if node.iterables is not None]
    inodes_no_src = [node for node in inodes if not node.itersource]
    inodes_src = [node for node in inodes if node.itersource]
    inodes_no_src.reverse()
    return inodes_no_src + inodes_src