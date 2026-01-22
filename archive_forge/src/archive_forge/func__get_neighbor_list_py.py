from __future__ import annotations
import collections
import contextlib
import functools
import inspect
import io
import itertools
import json
import math
import os
import random
import re
import sys
import warnings
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from inspect import isclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Callable, Literal, SupportsIndex, cast, get_args
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from numpy import cross, eye
from numpy.linalg import norm
from ruamel.yaml import YAML
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import expm, polar
from scipy.spatial.distance import squareform
from tabulate import tabulate
from pymatgen.core.bonds import CovalentBond, get_bond_length
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice, get_points_in_spheres
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import DummySpecies, Element, Species, get_el_sp
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.units import Length, Mass
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
from pymatgen.util.coord import all_distances, get_angle, lattice_points_in_supercell
def _get_neighbor_list_py(self, r: float, sites: list[PeriodicSite] | None=None, numerical_tol: float=1e-08, exclude_self: bool=True) -> tuple[np.ndarray, ...]:
    """A python version of getting neighbor_list. The returned values are a tuple of
        numpy arrays (center_indices, points_indices, offset_vectors, distances).
        Atom `center_indices[i]` has neighbor atom `points_indices[i]` that is
        translated by `offset_vectors[i]` lattice vectors, and the distance is
        `distances[i]`.

        Args:
            r (float): Radius of sphere
            sites (list of Sites or None): sites for getting all neighbors,
                default is None, which means neighbors will be obtained for all
                sites. This is useful in the situation where you are interested
                only in one subspecies type, and makes it a lot faster.
            numerical_tol (float): This is a numerical tolerance for distances.
                Sites which are < numerical_tol are determined to be coincident
                with the site. Sites which are r + numerical_tol away is deemed
                to be within r from the site. The default of 1e-8 should be
                ok in most instances.
            exclude_self (bool): whether to exclude atom neighboring with itself within
                numerical tolerance distance, default to True

        Returns:
            tuple: (center_indices, points_indices, offset_vectors, distances)
        """
    neighbors = self.get_all_neighbors_py(r=r, include_index=True, include_image=True, sites=sites, numerical_tol=1e-08)
    center_indices = []
    points_indices = []
    offsets = []
    distances = []
    for idx, nns in enumerate(neighbors):
        if len(nns) > 0:
            for nn in nns:
                if exclude_self and idx == nn.index and (nn.nn_distance <= numerical_tol):
                    continue
                center_indices.append(idx)
                points_indices.append(nn.index)
                offsets.append(nn.image)
                distances.append(nn.nn_distance)
    return tuple(map(np.array, (center_indices, points_indices, offsets, distances)))