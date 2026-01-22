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
class MapboxStyleTiles(GoogleWTS):
    """
    Implement web tile retrieval from a user-defined Mapbox style. For more
    details on Mapbox styles, see
    https://www.mapbox.com/studio-manual/overview/map-styling/.

    For terms of service, see https://www.mapbox.com/tos/.

    """

    def __init__(self, access_token, username, map_id, cache=False):
        """
        Set up a new instance to retrieve tiles from a Mapbox style.

        Access to Mapbox web services requires an access token and a map ID.
        See https://www.mapbox.com/api-documentation/ for details.

        Parameters
        ----------
        access_token
            A valid Mapbox API access token.
        username
            The username for the Mapbox user who defined the Mapbox style.
        map_id
            A map ID for a map defined by a Mapbox style. This is the map whose
            tiles will be retrieved through this process. Note that this style
            may be private and if your access token does not have permissions
            to view this style, then map tile retrieval will fail.

        """
        self.access_token = access_token
        self.username = username
        self.map_id = map_id
        super().__init__(cache=cache)

    def _image_url(self, tile):
        x, y, z = tile
        return f'https://api.mapbox.com/styles/v1/{self.username}/{self.map_id}/tiles/256/{z}/{x}/{y}?access_token={self.access_token}'