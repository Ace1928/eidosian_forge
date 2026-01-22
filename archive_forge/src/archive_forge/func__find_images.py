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
def _find_images(self, target_domain, target_z, start_tile=(0, 0, 0)):
    """Target domain is a shapely polygon in native coordinates."""
    assert isinstance(target_z, int) and target_z >= 0, 'target_z must be an integer >=0.'
    x0, x1, y0, y1 = self._tileextent(start_tile)
    domain = sgeom.box(x0, y0, x1, y1)
    if domain.intersects(target_domain):
        if start_tile[2] == target_z:
            yield start_tile
        else:
            for tile in self._subtiles(start_tile):
                yield from self._find_images(target_domain, target_z, start_tile=tile)