import datetime
import numpy as np
import shapely.geometry as sgeom
from .. import crs as ccrs
from . import ShapelyFeature

        Shade the darkside of the Earth, accounting for refraction.

        Parameters
        ----------
        date : datetime
            A UTC datetime object used to calculate the position of the sun.
            Default: datetime.datetime.utcnow()
        delta : float
            Stepsize in degrees to determine the resolution of the
            night polygon feature (``npts = 180 / delta``).
        refraction : float
            The adjustment in degrees due to refraction,
            thickness of the solar disc, elevation etc...

        Note
        ----
            Matplotlib keyword arguments can be used when drawing the feature.
            This allows standard Matplotlib control over aspects such as
            'color', 'alpha', etc.

        