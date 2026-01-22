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
def _get_ksymm_kpath(self, has_magmoms, magmom_axis, axis_specified, symprec, angle_tolerance, atol):
    ID = np.eye(3)
    PAR = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    V = self._latt.matrix.T
    W = self._rec_lattice.matrix.T
    A = np.dot(np.linalg.inv(W), V)
    if has_magmoms:
        grey_struct = self._structure.copy()
        grey_struct.remove_site_property('magmom')
        sga = SpacegroupAnalyzer(grey_struct, symprec=symprec, angle_tolerance=angle_tolerance)
        grey_ops = sga.get_symmetry_operations()
        self._structure = self._convert_all_magmoms_to_vectors(magmom_axis, axis_specified)
        mag_ops = self._get_magnetic_symmetry_operations(self._structure, grey_ops, atol)
        D = [SymmOp.from_rotation_and_translation(rotation_matrix=op.rotation_matrix, translation_vec=op.translation_vector) for op in mag_ops if op.time_reversal == 1]
        fD = [SymmOp.from_rotation_and_translation(rotation_matrix=op.rotation_matrix, translation_vec=op.translation_vector) for op in mag_ops if op.time_reversal == -1]
        if np.array([m == np.array([0, 0, 0]) for m in self._structure.site_properties['magmom']]).all():
            fD = D
            D = []
        if len(fD) == 0:
            self._mag_type = '1'
            isomorphic_point_group = [d.rotation_matrix for d in D]
            recip_point_group = self._get_reciprocal_point_group(isomorphic_point_group, ID, A)
        elif len(D) == 0:
            self._mag_type = '2'
            isomorphic_point_group = [d.rotation_matrix for d in fD]
            recip_point_group = self._get_reciprocal_point_group(isomorphic_point_group, PAR, A)
        else:
            self._mag_type = '3/4'
            f = self._get_coset_factor(D + fD, D)
            isomorphic_point_group = [d.rotation_matrix for d in D]
            recip_point_group = self._get_reciprocal_point_group(isomorphic_point_group, np.dot(PAR, f.rotation_matrix), A)
    else:
        self._mag_type = '0'
        if 'magmom' in self._structure.site_properties:
            warn('The parameter has_magmoms is False, but site_properties contains the key magmom.This property will be removed and could result in different symmetry operations.')
            self._structure.remove_site_property('magmom')
        sga = SpacegroupAnalyzer(self._structure)
        ops = sga.get_symmetry_operations()
        isomorphic_point_group = [op.rotation_matrix for op in ops]
        recip_point_group = self._get_reciprocal_point_group(isomorphic_point_group, PAR, A)
    self._rpg = recip_point_group
    key_points, bz_as_key_point_inds, face_center_inds = self._get_key_points()
    key_points_inds_orbits = self._get_key_point_orbits(key_points=key_points)
    key_lines = self._get_key_lines(key_points=key_points, bz_as_key_point_inds=bz_as_key_point_inds)
    key_lines_inds_orbits = self._get_key_line_orbits(key_points=key_points, key_lines=key_lines, key_points_inds_orbits=key_points_inds_orbits)
    little_groups_points, little_groups_lines = self._get_little_groups(key_points=key_points, key_points_inds_orbits=key_points_inds_orbits, key_lines_inds_orbits=key_lines_inds_orbits)
    _point_orbits_in_path, line_orbits_in_path = self._choose_path(key_points=key_points, key_points_inds_orbits=key_points_inds_orbits, key_lines_inds_orbits=key_lines_inds_orbits, little_groups_points=little_groups_points, little_groups_lines=little_groups_lines)
    IRBZ_points_inds = self._get_IRBZ(recip_point_group, W, key_points, face_center_inds, atol)
    lines_in_path_inds = []
    for ind in line_orbits_in_path:
        for tup in key_lines_inds_orbits[ind]:
            if tup[0] in IRBZ_points_inds and tup[1] in IRBZ_points_inds:
                lines_in_path_inds.append(tup)
                break
    G = nx.Graph(lines_in_path_inds)
    lines_in_path_inds = list(nx.edge_dfs(G))
    points_in_path_inds = [ind for tup in lines_in_path_inds for ind in tup]
    points_in_path_inds_unique = list(set(points_in_path_inds))
    orbit_cosines = []
    for orbit in key_points_inds_orbits[:-1]:
        current_orbit_cosines = []
        for orbit_index in orbit:
            key_point = key_points[orbit_index]
            for point_index in range(26):
                label_point = self.label_points(point_index)
                cosine_value = np.dot(key_point, label_point)
                cosine_value /= np.linalg.norm(key_point) * np.linalg.norm(label_point)
                cosine_value = np.round(cosine_value, decimals=3)
                current_orbit_cosines.append((point_index, cosine_value))
        sorted_cosines = sorted(current_orbit_cosines, key=lambda x: (-x[1], x[0]))
        orbit_cosines.append(sorted_cosines)
    orbit_labels = self._get_orbit_labels(orbit_cosines, key_points_inds_orbits, atol)
    key_points_labels = ['' for i in range(len(key_points))]
    for idx, orbit in enumerate(key_points_inds_orbits):
        for point_ind in orbit:
            key_points_labels[point_ind] = self.label_symbol(int(orbit_labels[idx]))
    kpoints = {}
    reverse_kpoints = {}
    for point_ind in points_in_path_inds_unique:
        point_label = key_points_labels[point_ind]
        if point_label not in kpoints:
            kpoints[point_label] = key_points[point_ind]
            reverse_kpoints[point_ind] = point_label
        else:
            existing_labels = [key for key in kpoints if point_label in key]
            if "'" not in point_label:
                existing_labels[:] = [label for label in existing_labels if "'" not in label]
            if len(existing_labels) == 1:
                max_occurence = 0
            elif "'" not in point_label:
                max_occurence = max((int(label[3:-1]) for label in existing_labels[1:]))
            else:
                max_occurence = max((int(label[4:-1]) for label in existing_labels[1:]))
            kpoints[f'{point_label}_{{{max_occurence + 1}}}'] = key_points[point_ind]
            reverse_kpoints[point_ind] = f'{point_label}_{{{max_occurence + 1}}}'
    path = []
    idx = 0
    start_of_subpath = True
    while idx < len(points_in_path_inds):
        if start_of_subpath:
            path.append([reverse_kpoints[points_in_path_inds[idx]]])
            idx += 1
            start_of_subpath = False
        elif points_in_path_inds[idx] == points_in_path_inds[idx + 1]:
            path[-1].append(reverse_kpoints[points_in_path_inds[idx]])
            idx += 2
        else:
            path[-1].append(reverse_kpoints[points_in_path_inds[idx]])
            idx += 1
            start_of_subpath = True
        if idx == len(points_in_path_inds) - 1:
            path[-1].append(reverse_kpoints[points_in_path_inds[idx]])
            idx += 1
    return {'kpoints': kpoints, 'path': path}