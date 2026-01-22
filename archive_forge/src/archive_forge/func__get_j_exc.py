from __future__ import annotations
import copy
import logging
from ast import literal_eval
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from monty.json import MSONable, jsanitize
from monty.serialization import dumpfn
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer, Ordering
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def _get_j_exc(self, i, j, dist):
    """
        Convenience method for looking up exchange parameter between two sites.

        Args:
            i (int): index of ith site
            j (int): index of jth site
            dist (float): distance (Angstrom) between sites +- tol

        Returns:
            j_exc (float): Exchange parameter in meV
        """
    for k in self.unique_site_ids:
        if i in k:
            i_index = self.unique_site_ids[k]
        if j in k:
            j_index = self.unique_site_ids[k]
    order = ''
    if abs(dist - self.dists['nn']) <= self.tol:
        order = '-nn'
    elif abs(dist - self.dists['nnn']) <= self.tol:
        order = '-nnn'
    elif abs(dist - self.dists['nnnn']) <= self.tol:
        order = '-nnnn'
    j_ij = f'{i_index}-{j_index}{order}'
    j_ji = f'{j_index}-{i_index}{order}'
    if j_ij in self.ex_params:
        j_exc = self.ex_params[j_ij]
    elif j_ji in self.ex_params:
        j_exc = self.ex_params[j_ji]
    else:
        j_exc = 0
    if '<J>' in self.ex_params and order == '-nn':
        j_exc = self.ex_params['<J>']
    return j_exc