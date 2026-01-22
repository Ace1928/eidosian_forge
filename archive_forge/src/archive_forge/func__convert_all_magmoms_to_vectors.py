from __future__ import annotations
import abc
import itertools
from math import ceil, cos, e, pi, sin, tan
from typing import TYPE_CHECKING, Any
from warnings import warn
import networkx as nx
import numpy as np
import spglib
from monty.dev import requires
from scipy.linalg import sqrtm
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import MagSymmOp, SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, cite_conventional_cell_algo
def _convert_all_magmoms_to_vectors(self, magmom_axis, axis_specified):
    struct = self._structure.copy()
    magmom_axis = np.array(magmom_axis)
    if 'magmom' not in struct.site_properties:
        warn("The 'magmom' property is not set in the structure's site properties.All magnetic moments are being set to zero.")
        struct.add_site_property('magmom', [np.array([0, 0, 0]) for _ in range(len(struct))])
        return struct
    old_magmoms = struct.site_properties['magmom']
    new_magmoms = []
    found_scalar = False
    for magmom in old_magmoms:
        if isinstance(magmom, np.ndarray):
            new_magmoms.append(magmom)
        elif isinstance(magmom, list):
            new_magmoms.append(np.array(magmom))
        else:
            found_scalar = True
            new_magmoms.append(magmom * magmom_axis)
    if found_scalar and (not axis_specified):
        warn('At least one magmom had a scalar value and magmom_axis was not specified. Defaulted to z+ spinor.')
    struct.remove_site_property('magmom')
    struct.add_site_property('magmom', new_magmoms)
    return struct