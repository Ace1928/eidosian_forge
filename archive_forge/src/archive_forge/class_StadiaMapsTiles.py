from abc import ABCMeta, abstractmethod
import concurrent.futures
import io
from pathlib import Path
import warnings
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
import cartopy
import cartopy.crs as ccrs
class StadiaMapsTiles(GoogleWTS):
    """
    Retrieves tiles from stadiamaps.com.

    For a full reference on the styles available please see
    https://docs.stadiamaps.com/themes/. A few of the specific styles
    that are made available are ``alidade_smooth``, ``stamen_terrain`` and
    ``osm_bright``.

    Using the Stadia Maps API requires including an attribution. Please see
    https://docs.stadiamaps.com/attribution/ for details.

    For most styles that means including the following attribution:

    `© Stadia Maps <https://www.stadiamaps.com/>`_
    `© OpenMapTiles <https://openmaptiles.org/>`_
    `© OpenStreetMap contributors <https://www.openstreetmap.org/about/>`_

    with Stamen styles *additionally* requiring the following attribution:

    `© Stamen Design <https://stamen.com/>`_

    Parameters
    ----------
    apikey : str, required
        The authentication key provided by Stadia Maps to query their APIs
    style : str, optional
        Name of the desired style. Defaults to ``alidade_smooth``.
        See https://docs.stadiamaps.com/themes/ for a full list of styles.
    resolution : str, optional
        Resolution of the images to return. Defaults to an empty string,
        standard resolution (256x256). You can also specify "@2x" for high
        resolution (512x512) tiles.
    cache : bool or str, optional
        If True, the default cache directory is used. If False, no cache is
        used. If a string, the string is used as the path to the cache.
    """

    def __init__(self, apikey, style='alidade_smooth', resolution='', cache=False):
        super().__init__(cache=cache, desired_tile_form='RGBA')
        self.apikey = apikey
        self.style = style
        self.resolution = resolution
        if style == 'stamen_watercolor':
            self.extension = 'jpg'
        else:
            self.extension = 'png'

    def _image_url(self, tile):
        x, y, z = tile
        return f'http://tiles.stadiamaps.com/tiles/{self.style}/{z}/{x}/{y}{self.resolution}.{self.extension}?api_key={self.apikey}'