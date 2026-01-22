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
def _image_url(self, tile):
    x, y, z = tile
    return f'https://tiles-a.koordinates.com/services;key={self.apikey}/tiles/{self.api_version}/layer={self.layer_id}/EPSG:3857/{z}/{x}/{y}.png'