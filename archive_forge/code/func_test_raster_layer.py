import numpy as np
import pytest
from holoviews.element import RGB, Bounds, Points, Tiles
from holoviews.element.tiles import _ATTRIBUTIONS, StamenTerrain
from .test_plot import TestPlotlyPlot, plotly_renderer
def test_raster_layer(self):
    tiles = StamenTerrain().redim.range(x=self.x_range, y=self.y_range).opts(alpha=0.7, min_zoom=3, max_zoom=7)
    fig_dict = plotly_renderer.get_plot_state(tiles)
    self.assertEqual(len(fig_dict['data']), 1)
    dummy_trace = fig_dict['data'][0]
    self.assertEqual(dummy_trace['type'], 'scattermapbox')
    self.assertEqual(dummy_trace['lon'], [])
    self.assertEqual(dummy_trace['lat'], [])
    self.assertEqual(dummy_trace['showlegend'], False)
    subplot = fig_dict['layout']['mapbox']
    self.assertEqual(subplot['style'], 'white-bg')
    self.assertEqual(subplot['center'], {'lat': self.lat_center, 'lon': self.lon_center})
    layers = fig_dict['layout']['mapbox'].get('layers', [])
    self.assertEqual(len(layers), 1)
    layer = layers[0]
    self.assertEqual(layer['source'][0].lower(), tiles.data.lower())
    self.assertEqual(layer['opacity'], 0.7)
    self.assertEqual(layer['sourcetype'], 'raster')
    self.assertEqual(layer['minzoom'], 3)
    self.assertEqual(layer['maxzoom'], 7)
    self.assertEqual(layer['sourceattribution'], _ATTRIBUTIONS['stamen', 'png'])