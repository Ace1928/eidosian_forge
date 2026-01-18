from __future__ import annotations
import json
import math
import os
import warnings
from bisect import bisect_left
from collections import defaultdict, namedtuple
from copy import deepcopy
from functools import lru_cache
from math import acos, asin, atan2, cos, exp, fabs, pi, pow, sin, sqrt
from typing import TYPE_CHECKING, Any, Literal, get_args
import numpy as np
from monty.dev import deprecated, requires
from monty.serialization import loadfn
from ruamel.yaml import YAML
from scipy.spatial import Voronoi
from pymatgen.analysis.bond_valence import BV_PARAMS, BVAnalyzer
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core import Element, IStructure, PeriodicNeighbor, PeriodicSite, Site, Species, Structure
@staticmethod
def transform_to_length(nn_data, length):
    """
        Given NNData, transforms data to the specified fingerprint length

        Args:
            nn_data: (NNData)
            length: (int) desired length of NNData.
        """
    if length is None:
        return nn_data
    if length:
        for cn in range(length):
            if cn not in nn_data.cn_weights:
                nn_data.cn_weights[cn] = 0
                nn_data.cn_nninfo[cn] = []
    return nn_data