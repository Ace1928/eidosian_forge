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
def _image_and_extent(self, wms_proj, wms_srs, wms_extent, output_proj, output_extent, target_resolution):
    min_x, max_x, min_y, max_y = wms_extent
    wms_image = self.service.getmap(layers=self.layers, srs=wms_srs, bbox=(min_x, min_y, max_x, max_y), size=target_resolution, format='image/png', **self.getmap_extra_kwargs)
    wms_image = Image.open(io.BytesIO(wms_image.read()))
    return _warped_located_image(wms_image, wms_proj, wms_extent, output_proj, output_extent, target_resolution)