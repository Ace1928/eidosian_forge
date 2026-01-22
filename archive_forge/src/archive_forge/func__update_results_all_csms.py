from __future__ import annotations
import itertools
import logging
import time
from random import shuffle
from typing import TYPE_CHECKING
import numpy as np
from numpy.linalg import norm, svd
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import (
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import (
from pymatgen.core import Lattice, Species, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
def _update_results_all_csms(self, result_dict, permutations, imin, geometry):
    permutation = permutations[imin]
    pdist = self.local_geometry.points_wocs_ctwocc(permutation=permutation)
    pperf = self.perfect_geometry.points_wocs_ctwocc()
    sm_info = symmetry_measure(points_distorted=pdist, points_perfect=pperf)
    result_dict[geometry.mp_symbol]['csm_wocs_ctwocc'] = sm_info['symmetry_measure']
    result_dict[geometry.mp_symbol]['rotation_matrix_wocs_ctwocc'] = np.linalg.inv(sm_info['rotation_matrix'])
    result_dict[geometry.mp_symbol]['scaling_factor_wocs_ctwocc'] = 1.0 / sm_info['scaling_factor']
    result_dict[geometry.mp_symbol]['translation_vector_wocs_ctwocc'] = self.local_geometry.centroid_without_centre
    pdist = self.local_geometry.points_wocs_ctwcc(permutation=permutation)
    pperf = self.perfect_geometry.points_wocs_ctwcc()
    sm_info = symmetry_measure(points_distorted=pdist, points_perfect=pperf)
    result_dict[geometry.mp_symbol]['csm_wocs_ctwcc'] = sm_info['symmetry_measure']
    result_dict[geometry.mp_symbol]['rotation_matrix_wocs_ctwcc'] = np.linalg.inv(sm_info['rotation_matrix'])
    result_dict[geometry.mp_symbol]['scaling_factor_wocs_ctwcc'] = 1.0 / sm_info['scaling_factor']
    result_dict[geometry.mp_symbol]['translation_vector_wocs_ctwcc'] = self.local_geometry.centroid_with_centre
    pdist = self.local_geometry.points_wocs_csc(permutation=permutation)
    pperf = self.perfect_geometry.points_wocs_csc()
    sm_info = symmetry_measure(points_distorted=pdist, points_perfect=pperf)
    result_dict[geometry.mp_symbol]['csm_wocs_csc'] = sm_info['symmetry_measure']
    result_dict[geometry.mp_symbol]['rotation_matrix_wocs_csc'] = np.linalg.inv(sm_info['rotation_matrix'])
    result_dict[geometry.mp_symbol]['scaling_factor_wocs_csc'] = 1.0 / sm_info['scaling_factor']
    result_dict[geometry.mp_symbol]['translation_vector_wocs_csc'] = self.local_geometry.bare_centre
    pdist = self.local_geometry.points_wcs_ctwocc(permutation=permutation)
    pperf = self.perfect_geometry.points_wcs_ctwocc()
    sm_info = symmetry_measure(points_distorted=pdist, points_perfect=pperf)
    result_dict[geometry.mp_symbol]['csm_wcs_ctwocc'] = sm_info['symmetry_measure']
    result_dict[geometry.mp_symbol]['rotation_matrix_wcs_ctwocc'] = np.linalg.inv(sm_info['rotation_matrix'])
    result_dict[geometry.mp_symbol]['scaling_factor_wcs_ctwocc'] = 1.0 / sm_info['scaling_factor']
    result_dict[geometry.mp_symbol]['translation_vector_wcs_ctwocc'] = self.local_geometry.centroid_without_centre
    pdist = self.local_geometry.points_wcs_ctwcc(permutation=permutation)
    pperf = self.perfect_geometry.points_wcs_ctwcc()
    sm_info = symmetry_measure(points_distorted=pdist, points_perfect=pperf)
    result_dict[geometry.mp_symbol]['csm_wcs_ctwcc'] = sm_info['symmetry_measure']
    result_dict[geometry.mp_symbol]['rotation_matrix_wcs_ctwcc'] = np.linalg.inv(sm_info['rotation_matrix'])
    result_dict[geometry.mp_symbol]['scaling_factor_wcs_ctwcc'] = 1.0 / sm_info['scaling_factor']
    result_dict[geometry.mp_symbol]['translation_vector_wcs_ctwcc'] = self.local_geometry.centroid_with_centre
    pdist = self.local_geometry.points_wcs_csc(permutation=permutation)
    pperf = self.perfect_geometry.points_wcs_csc()
    sm_info = symmetry_measure(points_distorted=pdist, points_perfect=pperf)
    result_dict[geometry.mp_symbol]['csm_wcs_csc'] = sm_info['symmetry_measure']
    result_dict[geometry.mp_symbol]['rotation_matrix_wcs_csc'] = np.linalg.inv(sm_info['rotation_matrix'])
    result_dict[geometry.mp_symbol]['scaling_factor_wcs_csc'] = 1.0 / sm_info['scaling_factor']
    result_dict[geometry.mp_symbol]['translation_vector_wcs_csc'] = self.local_geometry.bare_centre