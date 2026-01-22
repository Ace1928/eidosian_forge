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
class WMTSRasterSource(RasterSource):
    """
    A WMTS imagery retriever which can be added to a map.

    Uses tile caching for fast repeated map retrievals.

    Note
    ----
        Requires owslib and Pillow to work.

    """
    _shared_image_cache = weakref.WeakKeyDictionary()
    '\n    A nested mapping from WMTS, layer name, tile matrix name, tile row\n    and tile column to the resulting PIL image::\n\n        {wmts: {(layer_name, tile_matrix_name): {(row, column): Image}}}\n\n    This provides a significant boost when producing multiple maps of the\n    same projection or with an interactive figure.\n\n    '

    def __init__(self, wmts, layer_name, gettile_extra_kwargs=None):
        """
        Parameters
        ----------
        wmts
            The URL of the WMTS, or an owslib.wmts.WebMapTileService instance.
        layer_name
            The name of the layer to use.
        gettile_extra_kwargs: dict, optional
            Extra keywords (e.g. time) to pass through to the
            service's gettile method.

        """
        if WebMapService is None:
            raise ImportError(_OWSLIB_REQUIRED)
        if not (hasattr(wmts, 'tilematrixsets') and hasattr(wmts, 'contents') and hasattr(wmts, 'gettile')):
            wmts = owslib.wmts.WebMapTileService(wmts)
        try:
            layer = wmts.contents[layer_name]
        except KeyError:
            raise ValueError(f'Invalid layer name {layer_name!r} for WMTS at {wmts.url!r}')
        self.wmts = wmts
        self.layer = layer
        if gettile_extra_kwargs is None:
            gettile_extra_kwargs = {}
        self.gettile_extra_kwargs = gettile_extra_kwargs
        self._matrix_set_name_map = {}

    def _matrix_set_name(self, target_projection):
        key = id(target_projection)
        matrix_set_name = self._matrix_set_name_map.get(key)
        if matrix_set_name is None:
            if hasattr(self.layer, 'tilematrixsetlinks'):
                matrix_set_names = self.layer.tilematrixsetlinks.keys()
            else:
                matrix_set_names = self.layer.tilematrixsets

            def find_projection(match_projection):
                result = None
                for tile_matrix_set_name in matrix_set_names:
                    matrix_sets = self.wmts.tilematrixsets
                    tile_matrix_set = matrix_sets[tile_matrix_set_name]
                    crs_urn = tile_matrix_set.crs
                    tms_crs = None
                    if crs_urn in _URN_TO_CRS:
                        tms_crs = _URN_TO_CRS.get(crs_urn)
                    elif ':EPSG:' in crs_urn:
                        epsg_num = crs_urn.split(':')[-1]
                        tms_crs = ccrs.epsg(int(epsg_num))
                    if tms_crs == match_projection:
                        result = tile_matrix_set_name
                        break
                return result
            matrix_set_name = find_projection(target_projection)
            if matrix_set_name is None:
                for possible_projection in _URN_TO_CRS.values():
                    matrix_set_name = find_projection(possible_projection)
                    if matrix_set_name is not None:
                        break
                if matrix_set_name is None:
                    available_urns = sorted({self.wmts.tilematrixsets[name].crs for name in matrix_set_names})
                    msg = 'Unable to find tile matrix for projection.'
                    msg += f'\n    Projection: {target_projection}'
                    msg += '\n    Available tile CRS URNs:'
                    msg += '\n        ' + '\n        '.join(available_urns)
                    raise ValueError(msg)
            self._matrix_set_name_map[key] = matrix_set_name
        return matrix_set_name

    def validate_projection(self, projection):
        self._matrix_set_name(projection)

    def fetch_raster(self, projection, extent, target_resolution):
        matrix_set_name = self._matrix_set_name(projection)
        crs_urn = self.wmts.tilematrixsets[matrix_set_name].crs
        if crs_urn in _URN_TO_CRS:
            wmts_projection = _URN_TO_CRS[crs_urn]
        elif ':EPSG:' in crs_urn:
            epsg_num = crs_urn.split(':')[-1]
            wmts_projection = ccrs.epsg(int(epsg_num))
        else:
            raise ValueError(f'Unknown coordinate reference system string: {crs_urn}')
        if wmts_projection == projection:
            wmts_extents = [extent]
        else:
            wmts_extents = _target_extents(extent, projection, wmts_projection)
            resolution_factor = 1.4
            target_resolution = np.array(target_resolution) * resolution_factor
        width, height = target_resolution
        located_images = []
        for wmts_desired_extent in wmts_extents:
            min_x, max_x, min_y, max_y = wmts_desired_extent
            if wmts_projection == projection:
                max_pixel_span = min((max_x - min_x) / width, (max_y - min_y) / height)
            else:
                max_pixel_span = min(max_x - min_x, max_y - min_y) / max(width, height)
            wmts_image, wmts_actual_extent = self._wmts_images(self.wmts, self.layer, matrix_set_name, extent=wmts_desired_extent, max_pixel_span=max_pixel_span)
            if wmts_projection == projection:
                located_image = LocatedImage(wmts_image, wmts_actual_extent)
            else:
                located_image = _warped_located_image(wmts_image, wmts_projection, wmts_actual_extent, output_projection=projection, output_extent=extent, target_resolution=target_resolution)
            located_images.append(located_image)
        return located_images

    def _choose_matrix(self, tile_matrices, meters_per_unit, max_pixel_span):
        tile_matrices = sorted(tile_matrices, key=lambda tm: tm.scaledenominator, reverse=True)
        max_scale = max_pixel_span * meters_per_unit / METERS_PER_PIXEL
        for tm in tile_matrices:
            if tm.scaledenominator <= max_scale:
                return tm
        return tile_matrices[-1]

    def _tile_span(self, tile_matrix, meters_per_unit):
        pixel_span = tile_matrix.scaledenominator * (METERS_PER_PIXEL / meters_per_unit)
        tile_span_x = tile_matrix.tilewidth * pixel_span
        tile_span_y = tile_matrix.tileheight * pixel_span
        return (tile_span_x, tile_span_y)

    def _select_tiles(self, tile_matrix, tile_matrix_limits, tile_span_x, tile_span_y, extent):
        min_x, max_x, min_y, max_y = extent
        matrix_min_x, matrix_max_y = tile_matrix.topleftcorner
        epsilon = 1e-06
        min_col = int((min_x - matrix_min_x) / tile_span_x + epsilon)
        max_col = int((max_x - matrix_min_x) / tile_span_x - epsilon)
        min_row = int((matrix_max_y - max_y) / tile_span_y + epsilon)
        max_row = int((matrix_max_y - min_y) / tile_span_y - epsilon)
        min_col = max(min_col, 0)
        max_col = min(max_col, tile_matrix.matrixwidth - 1)
        min_row = max(min_row, 0)
        max_row = min(max_row, tile_matrix.matrixheight - 1)
        if tile_matrix_limits:
            min_col = max(min_col, tile_matrix_limits.mintilecol)
            max_col = min(max_col, tile_matrix_limits.maxtilecol)
            min_row = max(min_row, tile_matrix_limits.mintilerow)
            max_row = min(max_row, tile_matrix_limits.maxtilerow)
        return (min_col, max_col, min_row, max_row)

    def _wmts_images(self, wmts, layer, matrix_set_name, extent, max_pixel_span):
        """
        Add images from the specified WMTS layer and matrix set to cover
        the specified extent at an appropriate resolution.

        The zoom level (aka. tile matrix) is chosen to give the lowest
        possible resolution which still provides the requested quality.
        If insufficient resolution is available, the highest available
        resolution is used.

        Parameters
        ----------
        wmts
            The owslib.wmts.WebMapTileService providing the tiles.
        layer
            The owslib.wmts.ContentMetadata (aka. layer) to draw.
        matrix_set_name
            The name of the matrix set to use.
        extent
            Tuple of (left, right, bottom, top) in Axes coordinates.
        max_pixel_span
            Preferred maximum pixel width or height in Axes coordinates.

        """
        tile_matrix_set = wmts.tilematrixsets[matrix_set_name]
        tile_matrices = tile_matrix_set.tilematrix.values()
        if tile_matrix_set.crs in METERS_PER_UNIT:
            meters_per_unit = METERS_PER_UNIT[tile_matrix_set.crs]
        elif ':EPSG:' in tile_matrix_set.crs:
            epsg_num = tile_matrix_set.crs.split(':')[-1]
            tms_crs = ccrs.epsg(int(epsg_num))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                crs_dict = tms_crs.to_dict()
                crs_units = crs_dict.get('units', '')
                if crs_units != 'm':
                    raise ValueError('Only unit "m" implemented for EPSG projections.')
                meters_per_unit = 1
        else:
            raise ValueError(f'Unknown coordinate reference system string: {tile_matrix_set.crs}')
        tile_matrix = self._choose_matrix(tile_matrices, meters_per_unit, max_pixel_span)
        tile_span_x, tile_span_y = self._tile_span(tile_matrix, meters_per_unit)
        tile_matrix_set_links = getattr(layer, 'tilematrixsetlinks', None)
        if tile_matrix_set_links is None:
            tile_matrix_limits = None
        else:
            tile_matrix_set_link = tile_matrix_set_links[matrix_set_name]
            tile_matrix_limits = tile_matrix_set_link.tilematrixlimits.get(tile_matrix.identifier)
        min_col, max_col, min_row, max_row = self._select_tiles(tile_matrix, tile_matrix_limits, tile_span_x, tile_span_y, extent)
        tile_matrix_id = tile_matrix.identifier
        cache_by_wmts = WMTSRasterSource._shared_image_cache
        cache_by_layer_matrix = cache_by_wmts.setdefault(wmts, {})
        image_cache = cache_by_layer_matrix.setdefault((layer.id, tile_matrix_id), {})
        big_img = None
        n_rows = 1 + max_row - min_row
        n_cols = 1 + max_col - min_col
        ignore_out_of_range = tile_matrix_set_links is None
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                img_key = (row, col)
                img = image_cache.get(img_key)
                if img is None:
                    try:
                        tile = wmts.gettile(layer=layer.id, tilematrixset=matrix_set_name, tilematrix=str(tile_matrix_id), row=str(row), column=str(col), **self.gettile_extra_kwargs)
                    except owslib.util.ServiceException as exception:
                        if 'TileOutOfRange' in exception.message and ignore_out_of_range:
                            continue
                        raise exception
                    img = Image.open(io.BytesIO(tile.read()))
                    image_cache[img_key] = img
                if big_img is None:
                    size = (img.size[0] * n_cols, img.size[1] * n_rows)
                    big_img = Image.new('RGBA', size, (255, 255, 255, 255))
                top = (row - min_row) * tile_matrix.tileheight
                left = (col - min_col) * tile_matrix.tilewidth
                big_img.paste(img, (left, top))
        if big_img is None:
            img_extent = None
        else:
            matrix_min_x, matrix_max_y = tile_matrix.topleftcorner
            min_img_x = matrix_min_x + tile_span_x * min_col
            max_img_y = matrix_max_y - tile_span_y * min_row
            img_extent = (min_img_x, min_img_x + n_cols * tile_span_x, max_img_y - n_rows * tile_span_y, max_img_y)
        return (big_img, img_extent)