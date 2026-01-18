import numpy as np
import pyproj
import shapely.geometry as sgeom

        Return the distance (in physical meters) of the given Shapely geometry.

        The geometry is assumed to be in spherical (lon, lat) coordinates.

        Parameters
        ----------
        geometry : `shapely.geometry.BaseGeometry`
            The Shapely geometry to compute the length of. For polygons, the
            exterior length will be calculated. For multi-part geometries, the
            sum of the parts will be computed.

        