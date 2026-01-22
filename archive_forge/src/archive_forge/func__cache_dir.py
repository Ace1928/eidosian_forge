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
@property
def _cache_dir(self):
    """Return the name of the cache directory"""
    return self.cache_path / self.__class__.__name__