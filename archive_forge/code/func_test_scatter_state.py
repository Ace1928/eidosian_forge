import numpy as np
from holoviews.element import Scatter, Tiles
from .test_plot import TestPlotlyPlot
def test_scatter_state(self):
    xs = [3000000, 2000000, 1000000]
    ys = [-3000000, -2000000, -1000000]
    x_range = (-5000000, 4000000)
    x_center = sum(x_range) / 2.0
    y_range = (-3000000, 2000000)
    y_center = sum(y_range) / 2.0
    lon_centers, lat_centers = Tiles.easting_northing_to_lon_lat([x_center], [y_center])
    lon_center, lat_center = (lon_centers[0], lat_centers[0])
    lons, lats = Tiles.easting_northing_to_lon_lat(xs, ys)
    scatter = Tiles('') * Scatter((xs, ys)).redim.range(x=x_range, y=y_range)
    state = self._get_plot_state(scatter)
    self.assertEqual(state['data'][1]['type'], 'scattermapbox')
    self.assertEqual(state['data'][1]['lon'], lons)
    self.assertEqual(state['data'][1]['lat'], lats)
    self.assertEqual(state['data'][1]['mode'], 'markers')
    self.assertEqual(state['layout']['mapbox']['center'], {'lat': lat_center, 'lon': lon_center})
    self.assertFalse('xaxis' in state['layout'])
    self.assertFalse('yaxis' in state['layout'])