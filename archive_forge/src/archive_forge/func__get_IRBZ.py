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
def _get_IRBZ(self, recip_point_group, W, key_points, face_center_inds, atol):
    rpgdict = self._get_reciprocal_point_group_dict(recip_point_group, atol)
    g = np.dot(W.T, W)
    g_inv = np.linalg.inv(g)
    D = np.linalg.det(W)
    primary_orientation = secondary_orientation = tertiary_orientation = None
    planar_boundaries = []
    IRBZ_points = list(enumerate(key_points))
    for sigma in rpgdict['reflections']:
        norm = sigma['normal']
        if primary_orientation is None:
            primary_orientation = norm
            planar_boundaries.append(norm)
        elif np.isclose(np.dot(primary_orientation, np.dot(g, norm)), 0, atol=atol):
            if secondary_orientation is None:
                secondary_orientation = norm
                planar_boundaries.append(norm)
            elif np.isclose(np.dot(secondary_orientation, np.dot(g, norm)), 0, atol=atol):
                if tertiary_orientation is None:
                    tertiary_orientation = norm
                    planar_boundaries.append(norm)
                elif np.allclose(norm, -1 * tertiary_orientation, atol=atol):
                    pass
            elif np.dot(secondary_orientation, np.dot(g, norm)) < 0:
                planar_boundaries.append(-1 * norm)
            else:
                planar_boundaries.append(norm)
        elif np.dot(primary_orientation, np.dot(g, norm)) < 0:
            planar_boundaries.append(-1 * norm)
        else:
            planar_boundaries.append(norm)
    IRBZ_points = self._reduce_IRBZ(IRBZ_points, planar_boundaries, g, atol)
    used_axes = []
    for rotn in rpgdict['rotations']['six-fold']:
        ax = rotn['axis']
        op = rotn['op']
        if not np.any([np.allclose(ax, usedax, atol) for usedax in used_axes]) and self._op_maps_IRBZ_to_self(op, IRBZ_points, atol):
            face_center_found = False
            for point in IRBZ_points:
                if point[0] in face_center_inds:
                    cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                    if not np.allclose(cross, 0, atol=atol):
                        rot_boundaries = [cross, -1 * np.dot(op, cross)]
                        face_center_found = True
                        used_axes.append(ax)
                        break
            if not face_center_found:
                print('face center not found')
                for point in IRBZ_points:
                    cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                    if not np.allclose(cross, 0, atol=atol):
                        rot_boundaries = [cross, -1 * np.dot(op, cross)]
                        used_axes.append(ax)
                        break
            IRBZ_points = self._reduce_IRBZ(IRBZ_points, rot_boundaries, g, atol)
    for rotn in rpgdict['rotations']['rotoinv-four-fold']:
        ax = rotn['axis']
        op = rotn['op']
        if not np.any([np.allclose(ax, usedax, atol) for usedax in used_axes]) and self._op_maps_IRBZ_to_self(op, IRBZ_points, atol):
            face_center_found = False
            for point in IRBZ_points:
                if point[0] in face_center_inds:
                    cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                    if not np.allclose(cross, 0, atol=atol):
                        rot_boundaries = [cross, np.dot(op, cross)]
                        face_center_found = True
                        used_axes.append(ax)
                        break
            if not face_center_found:
                print('face center not found')
                for point in IRBZ_points:
                    cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                    if not np.allclose(cross, 0, atol=atol):
                        rot_boundaries = [cross, -1 * np.dot(op, cross)]
                        used_axes.append(ax)
                        break
            IRBZ_points = self._reduce_IRBZ(IRBZ_points, rot_boundaries, g, atol)
    for rotn in rpgdict['rotations']['four-fold']:
        ax = rotn['axis']
        op = rotn['op']
        if not np.any([np.allclose(ax, usedax, atol) for usedax in used_axes]) and self._op_maps_IRBZ_to_self(op, IRBZ_points, atol):
            face_center_found = False
            for point in IRBZ_points:
                if point[0] in face_center_inds:
                    cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                    if not np.allclose(cross, 0, atol=atol):
                        rot_boundaries = [cross, -1 * np.dot(op, cross)]
                        face_center_found = True
                        used_axes.append(ax)
                        break
            if not face_center_found:
                print('face center not found')
                for point in IRBZ_points:
                    cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                    if not np.allclose(cross, 0, atol=atol):
                        rot_boundaries = [cross, -1 * np.dot(op, cross)]
                        used_axes.append(ax)
                        break
            IRBZ_points = self._reduce_IRBZ(IRBZ_points, rot_boundaries, g, atol)
    for rotn in rpgdict['rotations']['rotoinv-three-fold']:
        ax = rotn['axis']
        op = rotn['op']
        if not np.any([np.allclose(ax, usedax, atol) for usedax in used_axes]) and self._op_maps_IRBZ_to_self(op, IRBZ_points, atol):
            face_center_found = False
            for point in IRBZ_points:
                if point[0] in face_center_inds:
                    cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                    if not np.allclose(cross, 0, atol=atol):
                        rot_boundaries = [cross, -1 * np.dot(sqrtm(-1 * op), cross)]
                        face_center_found = True
                        used_axes.append(ax)
                        break
            if not face_center_found:
                print('face center not found')
                for point in IRBZ_points:
                    cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                    if not np.allclose(cross, 0, atol=atol):
                        rot_boundaries = [cross, -1 * np.dot(op, cross)]
                        used_axes.append(ax)
                        break
            IRBZ_points = self._reduce_IRBZ(IRBZ_points, rot_boundaries, g, atol)
    for rotn in rpgdict['rotations']['three-fold']:
        ax = rotn['axis']
        op = rotn['op']
        if not np.any([np.allclose(ax, usedax, atol) for usedax in used_axes]) and self._op_maps_IRBZ_to_self(op, IRBZ_points, atol):
            face_center_found = False
            for point in IRBZ_points:
                if point[0] in face_center_inds:
                    cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                    if not np.allclose(cross, 0, atol=atol):
                        rot_boundaries = [cross, -1 * np.dot(op, cross)]
                        face_center_found = True
                        used_axes.append(ax)
                        break
            if not face_center_found:
                print('face center not found')
                for point in IRBZ_points:
                    cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                    if not np.allclose(cross, 0, atol=atol):
                        rot_boundaries = [cross, -1 * np.dot(op, cross)]
                        used_axes.append(ax)
                        break
            IRBZ_points = self._reduce_IRBZ(IRBZ_points, rot_boundaries, g, atol)
    for rotn in rpgdict['rotations']['two-fold']:
        ax = rotn['axis']
        op = rotn['op']
        if not np.any([np.allclose(ax, usedax, atol) for usedax in used_axes]) and self._op_maps_IRBZ_to_self(op, IRBZ_points, atol):
            face_center_found = False
            for point in IRBZ_points:
                if point[0] in face_center_inds:
                    cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                    if not np.allclose(cross, 0, atol=atol):
                        rot_boundaries = [cross, -1 * np.dot(op, cross)]
                        face_center_found = True
                        used_axes.append(ax)
                        break
            if not face_center_found:
                print('face center not found')
                for point in IRBZ_points:
                    cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                    if not np.allclose(cross, 0, atol=atol):
                        rot_boundaries = [cross, -1 * np.dot(op, cross)]
                        used_axes.append(ax)
                        break
            IRBZ_points = self._reduce_IRBZ(IRBZ_points, rot_boundaries, g, atol)
    return [point[0] for point in IRBZ_points]