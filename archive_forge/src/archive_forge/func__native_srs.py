import collections
import io
import math
from urllib.parse import urlparse
import warnings
import weakref
from xml.etree import ElementTree
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
import cartopy.crs as ccrs
from cartopy.img_transform import warp_array
from cartopy.io import LocatedImage, RasterSource
def _native_srs(self, projection):
    native_srs_list = _CRS_TO_OGC_SRS.get(projection, None)
    if native_srs_list is None:
        return None
    else:
        contents = self.service.contents
        for native_srs in native_srs_list:
            native_OK = all((native_srs.lower() in map(str.lower, contents[layer].crsOptions) for layer in self.layers))
            if native_OK:
                return native_srs
        return None