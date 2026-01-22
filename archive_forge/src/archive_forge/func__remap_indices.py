from __future__ import annotations
import logging
import os
import subprocess
import warnings
from enum import Enum, unique
from glob import glob
from shutil import which
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from monty.serialization import loadfn
from monty.tempfile import ScratchDir
from scipy.spatial import KDTree
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import DummySpecies
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar, VolumetricData
from pymatgen.util.due import Doi, due
def _remap_indices(self):
    """Re-maps indices on self.nodes and self.edges such that node indices match
        that of structure, and then sorts self.nodes by index.
        """
    node_mapping = {}
    frac_coords = np.array(self.structure.frac_coords) % 1
    kd = KDTree(frac_coords)
    node_mapping = {}
    for idx, node in self.nodes.items():
        if self.critical_points[node['unique_idx']].type == CriticalPointType.nucleus:
            node_mapping[idx] = kd.query(node['frac_coords'])[1]
    if len(node_mapping) != len(self.structure):
        warnings.warn(f'Check that all sites in input structure ({len(self.structure)}) have been detected by critic2 ({len(node_mapping)}).')
    self.nodes = {node_mapping.get(idx, idx): node for idx, node in self.nodes.items()}
    for edge in self.edges.values():
        edge['from_idx'] = node_mapping.get(edge['from_idx'], edge['from_idx'])
        edge['to_idx'] = node_mapping.get(edge['to_idx'], edge['to_idx'])