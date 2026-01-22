import collections
import contextlib
import functools
import json
import os
from pathlib import Path
import warnings
import weakref
import matplotlib as mpl
import matplotlib.artist
import matplotlib.axes
import matplotlib.contour
from matplotlib.image import imread
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
import numpy as np
import numpy.ma as ma
import shapely.geometry as sgeom
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.mpl import _MPL_38
import cartopy.mpl.contour
import cartopy.mpl.feature_artist as feature_artist
import cartopy.mpl.geocollection
import cartopy.mpl.patch as cpatch
from cartopy.mpl.slippy_image_artist import SlippyImageArtist
class InterProjectionTransform(mtransforms.Transform):
    """
    Transform coordinates from the source_projection to
    the ``target_projection``.

    """
    input_dims = 2
    output_dims = 2
    is_separable = False
    has_inverse = True

    def __init__(self, source_projection, target_projection):
        """
        Create the transform object from the given projections.

        Parameters
        ----------
        source_projection
            A :class:`~cartopy.crs.CRS`.
        target_projection
            A :class:`~cartopy.crs.CRS`.

        """
        self.source_projection = source_projection
        self.target_projection = target_projection
        mtransforms.Transform.__init__(self)

    def __repr__(self):
        return f'< {self.__class__.__name__!s} {self.source_projection!s} -> {self.target_projection!s} >'

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            result = NotImplemented
        else:
            result = self.source_projection == other.source_projection and self.target_projection == other.target_projection
        return result

    def __ne__(self, other):
        return not self == other

    def transform_non_affine(self, xy):
        """
        Transform from source to target coordinates.

        Parameters
        ----------
        xy
            An (n,2) array of points in source coordinates.

        Returns
        -------
        x, y
            An (n,2) array of transformed points in target coordinates.

        """
        prj = self.target_projection
        if isinstance(xy, np.ndarray):
            return prj.transform_points(self.source_projection, xy[:, 0], xy[:, 1])[:, 0:2]
        else:
            x, y = xy
            x, y = prj.transform_point(x, y, self.source_projection)
            return (x, y)

    def transform_path_non_affine(self, src_path):
        """
        Transform from source to target coordinates.

        Cache results, so subsequent calls with the same *src_path* argument
        (and the same source and target projections) are faster.

        Parameters
        ----------
        src_path
            A Matplotlib :class:`~matplotlib.path.Path` object
            with vertices in source coordinates.

        Returns
        -------
        result
            A Matplotlib :class:`~matplotlib.path.Path` with vertices
            in target coordinates.

        """
        mapping = _PATH_TRANSFORM_CACHE.get(src_path)
        if mapping is not None:
            key = (self.source_projection, self.target_projection)
            result = mapping.get(key)
            if result is not None:
                return result
        new_vertices = self.target_projection.quick_vertices_transform(src_path.vertices, self.source_projection)
        if new_vertices is not None:
            if new_vertices is src_path.vertices:
                return src_path
            else:
                return mpath.Path(new_vertices, src_path.codes)
        if src_path.vertices.shape == (1, 2):
            return mpath.Path(self.transform(src_path.vertices))
        transformed_geoms = []
        geoms = cpatch.path_to_geos(src_path)
        for geom in geoms:
            proj_geom = self.target_projection.project_geometry(geom, self.source_projection)
            transformed_geoms.append(proj_geom)
        if not transformed_geoms:
            result = mpath.Path(np.empty([0, 2]))
        else:
            paths = cpatch.geos_to_path(transformed_geoms)
            if not paths:
                return mpath.Path(np.empty([0, 2]))
            points, codes = list(zip(*[cpatch.path_segments(path, curves=False, simplify=False) for path in paths]))
            result = mpath.Path(np.concatenate(points, 0), np.concatenate(codes))
        key = (self.source_projection, self.target_projection)
        if mapping is None:
            _PATH_TRANSFORM_CACHE[src_path] = {key: result}
        else:
            mapping[key] = result
        return result

    def inverted(self):
        """
        Returns
        -------
        InterProjectionTransform
            A Matplotlib :class:`~matplotlib.transforms.Transform`
            from target to source coordinates.

        """
        return InterProjectionTransform(self.target_projection, self.source_projection)