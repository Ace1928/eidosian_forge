from __future__ import annotations
import copy
import logging
import os.path
import subprocess
import warnings
from collections import defaultdict, namedtuple
from itertools import combinations
from operator import itemgetter
from shutil import which
from typing import TYPE_CHECKING, Any, Callable, cast
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
from monty.dev import deprecated
from monty.json import MSONable
from networkx.drawing.nx_agraph import write_dot
from networkx.readwrite import json_graph
from scipy.spatial import KDTree
from scipy.stats import describe
from pymatgen.core import Lattice, Molecule, PeriodicSite, Structure
from pymatgen.core.structure import FunctionalGroups
from pymatgen.util.coord import lattice_points_in_supercell
from pymatgen.vis.structure_vtk import EL_COLORS
@classmethod
def _edges_to_str(cls, g):
    header = 'from    to  to_image    '
    header_line = '----  ----  ------------'
    edge_weight_name = g.graph['edge_weight_name']
    if edge_weight_name:
        print_weights = ['weight']
        edge_label = g.graph['edge_weight_name']
        edge_weight_units = g.graph['edge_weight_units']
        if edge_weight_units:
            edge_label += f' ({edge_weight_units})'
        header += f'  {edge_label}'
        header_line += f'  {'-' * max([18, len(edge_label)])}'
    else:
        print_weights = False
    out = f'{header}\n{header_line}\n'
    edges = list(g.edges(data=True))
    edges.sort(key=itemgetter(0, 1))
    if print_weights:
        for u, v, data in edges:
            out += f'{u:4}  {v:4}  {data.get('to_jimage', (0, 0, 0))!s:12}  {data.get('weight', 0):.3e}\n'
    else:
        for u, v, data in edges:
            out += f'{u:4}  {v:4}  {data.get('to_jimage', (0, 0, 0))!s:12}\n'
    return out