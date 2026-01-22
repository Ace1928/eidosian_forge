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
def _load_cache(self):
    """Load the cache"""
    if self.cache_path is not None:
        cache_dir = self._cache_dir
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
            if self._default_cache:
                warnings.warn(f'Cartopy created the following directory to cache GoogleWTS tiles: {cache_dir}')
        self.cache = self.cache.union(set(cache_dir.iterdir()))