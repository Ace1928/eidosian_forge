from __future__ import annotations
import collections
import itertools
from math import acos, pi
from typing import TYPE_CHECKING
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from pymatgen.analysis.local_env import JmolNN, VoronoiNN
from pymatgen.core import Composition, Element, PeriodicSite, Species
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def analyze_structures(self, structures, step_freq=10, most_frequent_polyhedra=15):
    """
        Perform Voronoi analysis on a list of Structures.
        Note that this might take a significant amount of time depending on the
        size and number of structures.

        Args:
            structures (list): list of Structures
            cutoff (float: cutoff distance around an atom to search for
                neighbors
            step_freq (int): perform analysis every step_freq steps
            qhull_options (str): options to pass to qhull
            most_frequent_polyhedra (int): this many unique polyhedra with
                highest frequencies is stored.

        Returns:
            A list of tuples in the form (voronoi_index,frequency)
        """
    voro_dict = {}
    step = 0
    for structure in structures:
        step += 1
        if step % step_freq != 0:
            continue
        v = []
        for n in range(len(structure)):
            v.append(str(self.analyze(structure, n=n).view()))
        for voro in v:
            if voro in voro_dict:
                voro_dict[voro] += 1
            else:
                voro_dict[voro] = 1
    return sorted(voro_dict.items(), key=lambda x: (x[1], x[0]), reverse=True)[:most_frequent_polyhedra]