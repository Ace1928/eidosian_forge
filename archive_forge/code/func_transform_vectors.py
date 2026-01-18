from abc import ABCMeta
from collections import OrderedDict
from functools import lru_cache
import io
import json
import math
import warnings
import numpy as np
from pyproj import Transformer
from pyproj.exceptions import ProjError
import shapely.geometry as sgeom
from shapely.prepared import prep
import cartopy.trace
def transform_vectors(self, src_proj, x, y, u, v):
    """
        transform_vectors(src_proj, x, y, u, v)

        Transform the given vector components, with coordinates in the
        given source coordinate system (``src_proj``), to this coordinate
        system. The vector components must be given relative to the
        source projection's coordinate reference system (grid eastward and
        grid northward).

        Parameters
        ----------
        src_proj
            The :class:`CRS.Projection` that represents the coordinate system
            the vectors are defined in.
        x
            The x coordinates of the vectors in the source projection.
        y
            The y coordinates of the vectors in the source projection.
        u
            The grid-eastward components of the vectors.
        v
            The grid-northward components of the vectors.

        Note
        ----
            x, y, u and v may be 1 or 2 dimensional, but must all have matching
            shapes.

        Returns
        -------
            ut, vt: The transformed vector components.

        Note
        ----
           The algorithm used to transform vectors is an approximation
           rather than an exact transform, but the accuracy should be
           good enough for visualization purposes.

        """
    if not x.shape == y.shape == u.shape == v.shape:
        raise ValueError('x, y, u and v arrays must be the same shape')
    if x.ndim not in (1, 2):
        raise ValueError('x, y, u and v must be 1 or 2 dimensional')
    proj_xyz = self.transform_points(src_proj, x, y)
    target_x, target_y = (proj_xyz[..., 0], proj_xyz[..., 1])
    vector_magnitudes = (u ** 2 + v ** 2) ** 0.5
    vector_angles = np.arctan2(v, u)
    factor = 360000.0
    delta = (src_proj.x_limits[1] - src_proj.x_limits[0]) / factor
    x_perturbations = delta * np.cos(vector_angles)
    y_perturbations = delta * np.sin(vector_angles)
    proj_xyz = src_proj.transform_points(src_proj, x, y)
    source_x, source_y = (proj_xyz[..., 0], proj_xyz[..., 1])
    eps = 1e-09
    invalid_x = np.logical_or(source_x + x_perturbations < src_proj.x_limits[0] - eps, source_x + x_perturbations > src_proj.x_limits[1] + eps)
    if invalid_x.any():
        x_perturbations[invalid_x] *= -1
        y_perturbations[invalid_x] *= -1
    invalid_y = np.logical_or(source_y + y_perturbations < src_proj.y_limits[0] - eps, source_y + y_perturbations > src_proj.y_limits[1] + eps)
    if invalid_y.any():
        x_perturbations[invalid_y] *= -1
        y_perturbations[invalid_y] *= -1
    reversed_vectors = np.logical_xor(invalid_x, invalid_y)
    problem_points = np.logical_or(source_x + x_perturbations < src_proj.x_limits[0] - eps, source_x + x_perturbations > src_proj.x_limits[1] + eps)
    if problem_points.any():
        warnings.warn('Some vectors at source domain corners may not have been transformed correctly')
    proj_xyz = self.transform_points(src_proj, source_x + x_perturbations, source_y + y_perturbations)
    target_x_perturbed = proj_xyz[..., 0]
    target_y_perturbed = proj_xyz[..., 1]
    projected_angles = np.arctan2(target_y_perturbed - target_y, target_x_perturbed - target_x)
    if reversed_vectors.any():
        projected_angles[reversed_vectors] += np.pi
    projected_u = vector_magnitudes * np.cos(projected_angles)
    projected_v = vector_magnitudes * np.sin(projected_angles)
    return (projected_u, projected_v)