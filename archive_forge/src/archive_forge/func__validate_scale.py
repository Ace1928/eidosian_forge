from abc import ABCMeta, abstractmethod
import numpy as np
import shapely.geometry as sgeom
import cartopy.crs
import cartopy.io.shapereader as shapereader
def _validate_scale(self):
    if self.scale not in ('110m', '50m', '10m'):
        raise ValueError(f'{self.scale!r} is not a valid Natural Earth scale. Valid scales are "110m", "50m", and "10m".')