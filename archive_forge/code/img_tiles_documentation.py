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

        Set up a new instance to retrieve tiles from The LINZ
        aka. Land Information New Zealand

        Access to LINZ WMTS GetCapabilities requires an API key.
        Register yourself free in https://id.koordinates.com/signup/
        to gain access into the LINZ database.

        Parameters
        ----------
        apikey
            A valid LINZ API key specific for every users.
        layer_id
            A layer ID for a map. See the "Technical Details" lower down the
            "About" tab for each layer displayed in the LINZ data service.
        api_version
            API version to use. Defaults to v4 for now.

        