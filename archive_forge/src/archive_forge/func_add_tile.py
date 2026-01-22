from __future__ import annotations
import logging # isort:skip
from contextlib import contextmanager
from typing import (
import xyzservices
from ..core.enums import (
from ..core.properties import (
from ..core.property.struct import Optional
from ..core.property_mixins import ScalarFillProps, ScalarLineProps
from ..core.query import find
from ..core.validation import error, warning
from ..core.validation.errors import (
from ..core.validation.warnings import MISSING_RENDERERS
from ..model import Model
from ..util.strings import nice_join
from ..util.warnings import warn
from .annotations import Annotation, Legend, Title
from .axes import Axis
from .dom import HTML
from .glyphs import Glyph
from .grids import Grid
from .layouts import GridCommon, LayoutDOM
from .ranges import (
from .renderers import GlyphRenderer, Renderer, TileRenderer
from .scales import (
from .sources import ColumnarDataSource, ColumnDataSource, DataSource
from .tiles import TileSource, WMTSTileSource
from .tools import HoverTool, Tool, Toolbar
def add_tile(self, tile_source: TileSource | xyzservices.TileProvider | str, retina: bool=False, **kwargs: Any) -> TileRenderer:
    """ Adds new ``TileRenderer`` into ``Plot.renderers``

        Args:
            tile_source (TileSource, xyzservices.TileProvider, str) :
                A tile source instance which contain tileset configuration

            retina (bool) :
                Whether to use retina version of tiles (if available)

        Keyword Arguments:
            Additional keyword arguments are passed on as-is to the tile renderer

        Returns:
            TileRenderer : TileRenderer

        """
    if not isinstance(tile_source, TileSource):
        if isinstance(tile_source, xyzservices.TileProvider):
            selected_provider = tile_source
        elif isinstance(tile_source, str):
            tile_source = tile_source.lower()
            if tile_source == 'esri_imagery':
                tile_source = 'esri_worldimagery'
            if tile_source == 'osm':
                tile_source = 'openstreetmap_mapnik'
            if tile_source.startswith('stamen'):
                tile_source = f'stadia.{tile_source}'
            if 'retina' in tile_source:
                tile_source = tile_source.replace('retina', '')
                retina = True
            selected_provider = xyzservices.providers.query_name(tile_source)
        scale_factor = '@2x' if retina else None
        tile_source = WMTSTileSource(url=selected_provider.build_url(scale_factor=scale_factor), attribution=selected_provider.html_attribution, min_zoom=selected_provider.get('min_zoom', 0), max_zoom=selected_provider.get('max_zoom', 30))
    tile_renderer = TileRenderer(tile_source=tile_source, **kwargs)
    self.renderers.append(tile_renderer)
    return tile_renderer