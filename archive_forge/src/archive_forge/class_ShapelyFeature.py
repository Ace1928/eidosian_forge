from abc import ABCMeta, abstractmethod
import numpy as np
import shapely.geometry as sgeom
import cartopy.crs
import cartopy.io.shapereader as shapereader
class ShapelyFeature(Feature):
    """
    A class capable of drawing a collection of
    shapely geometries.

    """

    def __init__(self, geometries, crs, **kwargs):
        """
        Parameters
        ----------
        geometries
            A collection of shapely geometries.
        crs
            The cartopy CRS in which the provided geometries are defined.

        Other Parameters
        ----------------
        **kwargs
            Keyword arguments to be used when drawing this feature.

        """
        super().__init__(crs, **kwargs)
        if isinstance(geometries, sgeom.base.BaseGeometry):
            geometries = [geometries]
        self._geoms = tuple(geometries)

    def geometries(self):
        return iter(self._geoms)