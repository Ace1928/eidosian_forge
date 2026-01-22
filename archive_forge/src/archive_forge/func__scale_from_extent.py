from abc import ABCMeta, abstractmethod
import numpy as np
import shapely.geometry as sgeom
import cartopy.crs
import cartopy.io.shapereader as shapereader
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