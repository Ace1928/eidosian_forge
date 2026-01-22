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
@staticmethod
def _get_reciprocal_point_group_dict(recip_point_group, atol):
    PAR = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    dct = {'reflections': [], 'rotations': {'two-fold': [], 'three-fold': [], 'four-fold': [], 'six-fold': [], 'rotoinv-three-fold': [], 'rotoinv-four-fold': [], 'rotoinv-six-fold': []}, 'inversion': []}
    for idx, op in enumerate(recip_point_group):
        evals, evects = np.linalg.eig(op)
        tr = np.trace(op)
        det = np.linalg.det(op)
        if np.isclose(det, 1, atol=atol):
            if np.isclose(tr, 3, atol=atol):
                continue
            if np.isclose(tr, -1, atol=atol):
                for j in range(3):
                    if np.isclose(evals[j], 1, atol=atol):
                        ax = evects[:, j]
                dct['rotations']['two-fold'].append({'ind': idx, 'axis': ax, 'op': op})
            elif np.isclose(tr, 0, atol=atol):
                for j in range(3):
                    if np.isreal(evals[j]) and np.isclose(np.absolute(evals[j]), 1, atol=atol):
                        ax = evects[:, j]
                dct['rotations']['three-fold'].append({'ind': idx, 'axis': ax, 'op': op})
            elif np.isclose(tr, 1, atol=atol):
                for j in range(3):
                    if np.isreal(evals[j]) and np.isclose(np.absolute(evals[j]), 1, atol=atol):
                        ax = evects[:, j]
                dct['rotations']['four-fold'].append({'ind': idx, 'axis': ax, 'op': op})
            elif np.isclose(tr, 2, atol=atol):
                for j in range(3):
                    if np.isreal(evals[j]) and np.isclose(np.absolute(evals[j]), 1, atol=atol):
                        ax = evects[:, j]
                dct['rotations']['six-fold'].append({'ind': idx, 'axis': ax, 'op': op})
        if np.isclose(det, -1, atol=atol):
            if np.isclose(tr, -3, atol=atol):
                dct['inversion'].append({'ind': idx, 'op': PAR})
            elif np.isclose(tr, 1, atol=atol):
                for j in range(3):
                    if np.isclose(evals[j], -1, atol=atol):
                        norm = evects[:, j]
                dct['reflections'].append({'ind': idx, 'normal': norm, 'op': op})
            elif np.isclose(tr, 0, atol=atol):
                for j in range(3):
                    if np.isreal(evals[j]) and np.isclose(np.absolute(evals[j]), 1, atol=atol):
                        ax = evects[:, j]
                dct['rotations']['rotoinv-three-fold'].append({'ind': idx, 'axis': ax, 'op': op})
            elif np.isclose(tr, -1, atol=atol):
                for j in range(3):
                    if np.isreal(evals[j]) and np.isclose(np.absolute(evals[j]), 1, atol=atol):
                        ax = evects[:, j]
                dct['rotations']['rotoinv-four-fold'].append({'ind': idx, 'axis': ax, 'op': op})
            elif np.isclose(tr, -2, atol=atol):
                for j in range(3):
                    if np.isreal(evals[j]) and np.isclose(np.absolute(evals[j]), 1, atol=atol):
                        ax = evects[:, j]
                dct['rotations']['rotoinv-six-fold'].append({'ind': idx, 'axis': ax, 'op': op})
    return dct