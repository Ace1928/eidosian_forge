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
class WMSRasterSource(RasterSource):
    """
    A WMS imagery retriever which can be added to a map.

    Note
    ----
        Requires owslib and Pillow to work.

    No caching of retrieved maps is done with this WMSRasterSource.

    To reduce load on the WMS server it is encouraged to tile
    map requests and subsequently stitch them together to recreate
    a single raster, thus allowing for a more aggressive caching scheme,
    but this WMSRasterSource does not currently implement WMS tile
    fetching.

    Whilst not the same service, there is also a WMTSRasterSource which
    makes use of tiles and comes with built-in caching for fast repeated
    map retrievals.

    """

    def __init__(self, service, layers, getmap_extra_kwargs=None):
        """
        Parameters
        ----------
        service: string or WebMapService instance
            The WebMapService instance, or URL of a WMS service,
            from whence to retrieve the image.
        layers: string or list of strings
            The name(s) of layers to use from the WMS service.
        getmap_extra_kwargs: dict, optional
            Extra keywords to pass through to the service's getmap method.
            If None, a dictionary with ``{'transparent': True}`` will be
            defined.

        """
        if WebMapService is None:
            raise ImportError(_OWSLIB_REQUIRED)
        if isinstance(service, str):
            service = WebMapService(service)
        if isinstance(layers, str):
            layers = [layers]
        if getmap_extra_kwargs is None:
            getmap_extra_kwargs = {'transparent': True}
        if len(layers) == 0:
            raise ValueError('One or more layers must be defined.')
        for layer in layers:
            if layer not in service.contents:
                raise ValueError(f'The {layer!r} layer does not exist in this service.')
        self.service = service
        self.layers = layers
        self.getmap_extra_kwargs = getmap_extra_kwargs

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

    def _fallback_proj_and_srs(self):
        """
        Return a :class:`cartopy.crs.Projection` and corresponding
        SRS string in which the WMS service can supply the requested
        layers.

        """
        contents = self.service.contents
        for proj, srs_list in _CRS_TO_OGC_SRS.items():
            for srs in srs_list:
                srs_OK = all((srs.lower() in map(str.lower, contents[layer].crsOptions) for layer in self.layers))
                if srs_OK:
                    return (proj, srs)
        raise ValueError('The requested layers are not available in a known SRS.')

    def validate_projection(self, projection):
        if self._native_srs(projection) is None:
            self._fallback_proj_and_srs()

    def _image_and_extent(self, wms_proj, wms_srs, wms_extent, output_proj, output_extent, target_resolution):
        min_x, max_x, min_y, max_y = wms_extent
        wms_image = self.service.getmap(layers=self.layers, srs=wms_srs, bbox=(min_x, min_y, max_x, max_y), size=target_resolution, format='image/png', **self.getmap_extra_kwargs)
        wms_image = Image.open(io.BytesIO(wms_image.read()))
        return _warped_located_image(wms_image, wms_proj, wms_extent, output_proj, output_extent, target_resolution)

    def fetch_raster(self, projection, extent, target_resolution):
        target_resolution = [math.ceil(val) for val in target_resolution]
        wms_srs = self._native_srs(projection)
        if wms_srs is not None:
            wms_proj = projection
            wms_extents = [extent]
        else:
            wms_proj, wms_srs = self._fallback_proj_and_srs()
            wms_extents = _target_extents(extent, projection, wms_proj)
        located_images = []
        for wms_extent in wms_extents:
            located_images.append(self._image_and_extent(wms_proj, wms_srs, wms_extent, projection, extent, target_resolution))
        return located_images