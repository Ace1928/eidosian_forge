import unittest
from unittest.mock import MagicMock, patch
from dash._callback_context import CallbackContext
from holoviews import Bounds, DynamicMap, Scatter
from holoviews.plotting.plotly.dash import (
from holoviews.streams import BoundsXY, RangeXY, Selection1D
from .test_plot import TestPlotlyPlot
import plotly.io as pio
def test_simple_element(self):
    scatter = Scatter([0, 0])
    components = to_dash(self.app, [scatter])
    self.assertIsInstance(components, DashComponents)
    self.assertEqual(len(components.graphs), 1)
    self.assertEqual(len(components.kdims), 0)
    self.assertIsInstance(components.store, Store)
    self.assertEqual(len(components.resets), 0)
    fig = components.graphs[0].figure
    self.assertEqual(len(fig['data']), 1)
    self.assertEqual(fig['data'][0]['type'], 'scatter')