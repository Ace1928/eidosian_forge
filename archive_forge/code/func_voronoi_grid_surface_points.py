from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from monty.json import MontyDecoder, MSONable, jsanitize
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.chemenv_errors import ChemenvError
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.core import Element, PeriodicNeighbor, PeriodicSite, Species, Structure
def voronoi_grid_surface_points(self, additional_condition=1, other_origins='DO_NOTHING'):
    """
            Get the surface points in the Voronoi grid for this neighbor from the sources.
            The general shape of the points should look like a staircase such as in the following figure :

               ^
            0.0|
               |
               |      B----C
               |      |    |
               |      |    |
            a  |      k    D-------E
            n  |      |            |
            g  |      |            |
            l  |      |            |
            e  |      j            F----n---------G
               |      |                           |
               |      |                           |
               |      A----g-------h----i---------H
               |
               |
            1.0+------------------------------------------------->
              1.0              distance              2.0   ->+Inf

            Args:
                additional_condition: Additional condition for the neighbors.
                other_origins: What to do with sources that do not come from the Voronoi grid (e.g. "from hints").
            """
    src_list = []
    for src in self.sources:
        if src['origin'] == 'dist_ang_ac_voronoi':
            if src['ac'] != additional_condition:
                continue
            src_list.append(src)
        else:
            if other_origins == 'DO_NOTHING':
                continue
            raise NotImplementedError('Nothing implemented for other sources ...')
    if len(src_list) == 0:
        return None
    dists = [src['dp_dict']['min'] for src in src_list]
    angles = [src['ap_dict']['max'] for src in src_list]
    next_dists = [src['dp_dict']['next'] for src in src_list]
    next_angles = [src['ap_dict']['next'] for src in src_list]
    points_dict = {}
    p_dists = []
    pangs = []
    for idx in range(len(src_list)):
        if not any(np.isclose(p_dists, dists[idx])):
            p_dists.append(dists[idx])
        if not any(np.isclose(p_dists, next_dists[idx])):
            p_dists.append(next_dists[idx])
        if not any(np.isclose(pangs, angles[idx])):
            pangs.append(angles[idx])
        if not any(np.isclose(pangs, next_angles[idx])):
            pangs.append(next_angles[idx])
        d1_indices = np.argwhere(np.isclose(p_dists, dists[idx])).flatten()
        if len(d1_indices) != 1:
            raise ValueError('Distance parameter not found ...')
        d2_indices = np.argwhere(np.isclose(p_dists, next_dists[idx])).flatten()
        if len(d2_indices) != 1:
            raise ValueError('Distance parameter not found ...')
        a1_indices = np.argwhere(np.isclose(pangs, angles[idx])).flatten()
        if len(a1_indices) != 1:
            raise ValueError('Angle parameter not found ...')
        a2_indices = np.argwhere(np.isclose(pangs, next_angles[idx])).flatten()
        if len(a2_indices) != 1:
            raise ValueError('Angle parameter not found ...')
        id1 = d1_indices[0]
        id2 = d2_indices[0]
        ia1 = a1_indices[0]
        ia2 = a2_indices[0]
        for id_ia in [(id1, ia1), (id1, ia2), (id2, ia1), (id2, ia2)]:
            points_dict.setdefault(id_ia, 0)
            points_dict[id_ia] += 1
    new_pts = []
    for pt, pt_nb in points_dict.items():
        if pt_nb % 2 == 1:
            new_pts.append(pt)
    sorted_points = [(0, 0)]
    move_ap_index = True
    while True:
        last_pt = sorted_points[-1]
        if move_ap_index:
            idp = last_pt[0]
            iap = None
            for pt in new_pts:
                if pt[0] == idp and pt != last_pt:
                    iap = pt[1]
                    break
        else:
            idp = None
            iap = last_pt[1]
            for pt in new_pts:
                if pt[1] == iap and pt != last_pt:
                    idp = pt[0]
                    break
        if (idp, iap) == (0, 0):
            break
        if (idp, iap) in sorted_points:
            raise ValueError('Error sorting points ...')
        sorted_points.append((idp, iap))
        move_ap_index = not move_ap_index
    return [(p_dists[idp], pangs[iap]) for idp, iap in sorted_points]