import numpy as np
import pytest
from holoviews.element import RGB, Bounds, Points, Tiles
from holoviews.element.tiles import _ATTRIBUTIONS, StamenTerrain
from .test_plot import TestPlotlyPlot, plotly_renderer
def test_xyzservices_tileprovider(self):
    xyzservices = pytest.importorskip('xyzservices')
    osm = xyzservices.providers.OpenStreetMap.Mapnik
    tiles = Tiles(osm, name='xyzservices').redim.range(x=self.x_range, y=self.y_range)
    fig_dict = plotly_renderer.get_plot_state(tiles)
    layers = fig_dict['layout']['mapbox'].get('layers', [])
    self.assertEqual(len(layers), 1)
    layer = layers[0]
    self.assertEqual(layer['source'][0].lower(), osm.build_url(scale_factor='@2x'))
    self.assertEqual(layer['maxzoom'], osm.max_zoom)
    self.assertEqual(layer['sourceattribution'], osm.html_attribution)