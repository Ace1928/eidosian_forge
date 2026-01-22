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
class MapQuestOSM(GoogleWTS):

    def _image_url(self, tile):
        x, y, z = tile
        url = f'https://otile1.mqcdn.com/tiles/1.0.0/osm/{z}/{x}/{y}.jpg'
        mqdevurl = 'https://devblog.mapquest.com/2016/06/15/modernization-of-mapquest-results-in-changes-to-open-tile-access/'
        warnings.warn(f'{url} will require a log in and will likely fail. see {mqdevurl} for more details.')
        return url