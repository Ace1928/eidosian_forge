import io
from pathlib import Path
import warnings
import numpy as np
from cartopy import config
import cartopy.crs as ccrs
from cartopy.io import Downloader, LocatedImage, RasterSource, fh_getter
def add_shading(elevation, azimuth, altitude):
    """Add shading to SRTM elevation data, using azimuth and altitude
    of the sun.

    Parameters
    ----------
    elevation
        SRTM elevation data (in meters)
    azimuth
        Azimuth of the Sun (in degrees)
    altitude
        Altitude of the Sun (in degrees)

    Return shaded SRTM relief map.
    """
    azimuth = np.deg2rad(azimuth)
    altitude = np.deg2rad(altitude)
    x, y = np.gradient(elevation)
    slope = np.pi / 2 - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)
    shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - np.pi / 2 - aspect)
    return shaded