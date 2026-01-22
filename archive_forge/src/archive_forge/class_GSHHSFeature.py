from abc import ABCMeta, abstractmethod
import numpy as np
import shapely.geometry as sgeom
import cartopy.crs
import cartopy.io.shapereader as shapereader
class GSHHSFeature(Feature):
    """
    An interface to the GSHHS dataset.

    See https://www.ngdc.noaa.gov/mgg/shorelines/gshhs.html

    Parameters
    ----------
    scale
        The dataset scale. One of 'auto', 'coarse', 'low', 'intermediate',
        'high, or 'full' (default is 'auto').
    levels
        A list of integers 1-6 corresponding to the desired GSHHS feature
        levels to draw (default is [1] which corresponds to coastlines).

    Other Parameters
    ----------------
    **kwargs
        Keyword arguments to be used when drawing the feature. Defaults
        are edgecolor='black' and facecolor='none'.

    """
    _geometries_cache = {}
    '\n    A mapping from scale and level to GSHHS shapely geometry::\n\n        {(scale, level): geom}\n\n    This provides a performance boost when plotting in interactive mode or\n    instantiating multiple GSHHS artists, by reducing repeated file IO.\n\n    '

    def __init__(self, scale='auto', levels=None, **kwargs):
        super().__init__(cartopy.crs.PlateCarree(), **kwargs)
        if scale not in ('auto', 'a', 'coarse', 'c', 'low', 'l', 'intermediate', 'i', 'high', 'h', 'full', 'f'):
            raise ValueError(f'Unknown GSHHS scale {scale!r}.')
        self._scale = scale
        if levels is None:
            levels = [1]
        self._levels = set(levels)
        unknown_levels = self._levels.difference([1, 2, 3, 4, 5, 6])
        if unknown_levels:
            raise ValueError(f'Unknown GSHHS levels {unknown_levels!r}.')
        self._kwargs.setdefault('edgecolor', 'black')
        self._kwargs.setdefault('facecolor', 'none')

    def _scale_from_extent(self, extent):
        """
        Return the appropriate scale (e.g. 'i') for the given extent
        expressed in PlateCarree CRS.

        """
        scale = 'c'
        if extent is not None:
            scale_limits = (('c', 20.0), ('l', 10.0), ('i', 2.0), ('h', 0.5), ('f', 0.1))
            width = abs(extent[1] - extent[0])
            height = abs(extent[3] - extent[2])
            min_extent = min(width, height)
            if min_extent != 0:
                for scale, limit in scale_limits:
                    if min_extent > limit:
                        break
        return scale

    def geometries(self):
        return self.intersecting_geometries(extent=None)

    def intersecting_geometries(self, extent):
        if self._scale == 'auto':
            scale = self._scale_from_extent(extent)
        else:
            scale = self._scale[0]
        if extent is not None:
            extent_geom = sgeom.box(extent[0], extent[2], extent[1], extent[3])
        for level in self._levels:
            geoms = GSHHSFeature._geometries_cache.get((scale, level))
            if geoms is None:
                path = shapereader.gshhs(scale, level)
                geoms = tuple(shapereader.Reader(path).geometries())
                GSHHSFeature._geometries_cache[scale, level] = geoms
            for geom in geoms:
                if extent is None or extent_geom.intersects(geom):
                    yield geom