from __future__ import annotations
import itertools
import logging
import warnings
from typing import TYPE_CHECKING
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from pymatgen.core.structure import Structure
from pymatgen.util.coord import get_angle
from pymatgen.util.string import unicodeify_spacegroup
def _get_all_miller_e(self):
    """
        From self: get miller_list(unique_miller), e_surf_list and symmetry operations(symm_ops)
        according to lattice apply symm_ops to get all the miller index, then get normal, get
        all the facets functions for Wulff shape calculation: |normal| = 1, e_surf is plane's
        distance to (0, 0, 0), normal[0]x + normal[1]y + normal[2]z = e_surf.

        Returns:
            [WulffFacet]
        """
    all_hkl = []
    color_ind = self.color_ind
    planes = []
    recp = self.structure.lattice.reciprocal_lattice_crystallographic
    recp_symm_ops = self.lattice.get_recp_symmetry_operation(self.symprec)
    for i, (hkl, energy) in enumerate(zip(self.hkl_list, self.e_surf_list)):
        for op in recp_symm_ops:
            miller = tuple((int(x) for x in op.operate(hkl)))
            if miller not in all_hkl:
                all_hkl.append(miller)
                normal = recp.get_cartesian_coords(miller)
                normal /= np.linalg.norm(normal)
                normal_pt = [x * energy for x in normal]
                dual_pt = [x / energy for x in normal]
                color_plane = color_ind[divmod(i, len(color_ind))[1]]
                planes.append(WulffFacet(normal, energy, normal_pt, dual_pt, color_plane, i, hkl))
    planes.sort(key=lambda x: x.e_surf)
    return planes