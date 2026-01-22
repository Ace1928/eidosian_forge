import numpy as np
from holoviews.element import (
from .test_plot import TestPlotlyPlot
class TestMapboxBox(TestMapboxShape):

    def test_single_box(self):
        box = Tiles('') * Box(0, 0, (1000000, 2000000)).redim.range(x=self.x_range, y=self.y_range)
        x_box_range = [-500000, 500000]
        y_box_range = [-1000000, 1000000]
        lon_box_range, lat_box_range = Tiles.easting_northing_to_lon_lat(x_box_range, y_box_range)
        state = self._get_plot_state(box)
        self.assertEqual(state['data'][1]['type'], 'scattermapbox')
        self.assertEqual(state['data'][1]['mode'], 'lines')
        self.assertEqual(state['data'][1]['showlegend'], False)
        self.assertEqual(state['data'][1]['line']['color'], default_shape_color)
        self.assertEqual(state['data'][1]['lon'], np.array([lon_box_range[i] for i in (0, 0, 1, 1, 0)]))
        self.assertEqual(state['data'][1]['lat'], np.array([lat_box_range[i] for i in (0, 1, 1, 0, 0)]))
        self.assertEqual(state['layout']['mapbox']['center'], {'lat': self.lat_center, 'lon': self.lon_center})