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
class Stamen(GoogleWTS):
    """
    Retrieves tiles from maps.stamen.com. Styles include
    ``terrain-background``, ``terrain``, ``toner`` and ``watercolor``.

    For a full reference on the styles available please see
    http://maps.stamen.com. Of particular note are the sub-styles
    that are made available (e.g. ``terrain`` and ``terrain-background``).
    To determine the name of the particular [sub-]style you want,
    follow the link on http://maps.stamen.com to your desired style and
    observe the style name in the URL. Your style name will be in the
    form of: ``http://maps.stamen.com/{STYLE_NAME}/#9/37/-122``.

    Except otherwise noted, the Stamen map tile sets are copyright Stamen
    Design, under a Creative Commons Attribution (CC BY 3.0) license.

    Please see the attribution notice at http://maps.stamen.com on how to
    attribute this imagery.

    References
    ----------

     * http://mike.teczno.com/notes/osm-us-terrain-layer/background.html
     * http://maps.stamen.com/
     * https://wiki.openstreetmap.org/wiki/List_of_OSM_based_Services
     * https://github.com/migurski/DEM-Tools

    """

    def __init__(self, style='toner', desired_tile_form=None, cache=False):
        warnings.warn('The Stamen styles are no longer served by Stamen and are now served by Stadia Maps. Please use the StadiaMapsTiles class instead.')
        layer_config = {'terrain': {'extension': 'png', 'opaque': True}, 'terrain-background': {'extension': 'png', 'opaque': True}, 'terrain-labels': {'extension': 'png', 'opaque': False}, 'terrain-lines': {'extension': 'png', 'opaque': False}, 'toner-background': {'extension': 'png', 'opaque': True}, 'toner': {'extension': 'png', 'opaque': True}, 'toner-hybrid': {'extension': 'png', 'opaque': False}, 'toner-labels': {'extension': 'png', 'opaque': False}, 'toner-lines': {'extension': 'png', 'opaque': False}, 'toner-lite': {'extension': 'png', 'opaque': True}, 'watercolor': {'extension': 'jpg', 'opaque': True}}
        layer_info = layer_config.get(style, {'extension': '.png', 'opaque': True})
        if desired_tile_form is None:
            if layer_info['opaque']:
                desired_tile_form = 'RGB'
            else:
                desired_tile_form = 'RGBA'
        super().__init__(desired_tile_form=desired_tile_form, cache=cache)
        self.style = style
        self.extension = layer_info['extension']

    def _image_url(self, tile):
        x, y, z = tile
        return 'http://tile.stamen.com/' + f'{self.style}/{z}/{x}/{y}.{self.extension}'