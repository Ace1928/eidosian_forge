import unittest
from unittest.mock import MagicMock, patch
from dash._callback_context import CallbackContext
from holoviews import Bounds, DynamicMap, Scatter
from holoviews.plotting.plotly.dash import (
from holoviews.streams import BoundsXY, RangeXY, Selection1D
from .test_plot import TestPlotlyPlot
import plotly.io as pio
def test_selection1d_dynamic_map(self):
    scatter = Scatter([[0, 0], [1, 1], [2, 2]])
    selection1d = Selection1D(source=scatter)
    dmap = DynamicMap(lambda index: scatter.iloc[index].opts(size=len(index) + 1), streams=[selection1d])
    components = to_dash(self.app, [scatter, dmap], reset_button=True)
    self.assertIsInstance(components, DashComponents)
    self.assertEqual(len(components.graphs), 2)
    self.assertEqual(len(components.kdims), 0)
    self.assertIsInstance(components.store, Store)
    self.assertEqual(len(components.resets), 1)
    decorator_args = next(iter(self.app.callback.call_args_list[0]))
    outputs, inputs, states = decorator_args
    expected_outputs = [(g.id, 'figure') for g in components.graphs] + [(components.store.id, 'data')]
    self.assertEqual([(output.component_id, output.component_property) for output in outputs], expected_outputs)
    expected_inputs = [(g.id, prop) for g in components.graphs for prop in ['selectedData', 'relayoutData']] + [(components.resets[0].id, 'n_clicks')]
    self.assertEqual([(ip.component_id, ip.component_property) for ip in inputs], expected_inputs)
    expected_state = [(components.store.id, 'data')]
    self.assertEqual([(state.component_id, state.component_property) for state in states], expected_state)
    fig1 = components.graphs[0].figure
    fig2 = components.graphs[1].figure
    self.assertEqual(len(fig2['data']), 1)
    self.assertEqual(fig2['data'][0]['marker']['size'], 1)
    self.assertEqual(list(fig2['data'][0]['x']), [])
    self.assertEqual(list(fig2['data'][0]['y']), [])
    callback_fn = self.app.callback.return_value.call_args[0][0]
    store_value = encode_store_data({'streams': {id(selection1d): selection1d.contents}})
    with patch.object(CallbackContext, 'triggered', [{'prop_id': inputs[0].component_id + '.selectedData'}]):
        [fig1, fig2, new_store] = callback_fn({'points': [{'curveNumber': 0, 'pointNumber': 0, 'pointIndex': 0}, {'curveNumber': 0, 'pointNumber': 2, 'pointIndex': 2}]}, {}, 0, store_value)
    self.assertEqual(len(fig2['data']), 1)
    self.assertEqual(fig2['data'][0]['marker']['size'], 3)
    self.assertEqual(list(fig2['data'][0]['x']), [0, 2])
    self.assertEqual(list(fig2['data'][0]['y']), [0, 2])
    self.assertEqual(decode_store_data(new_store), {'streams': {id(selection1d): {'index': [0, 2]}}, 'kdims': {}})
    store = new_store
    with patch.object(CallbackContext, 'triggered', [{'prop_id': components.resets[0].id + '.n_clicks'}]):
        [fig1, fig2, new_store] = callback_fn({}, {}, 1, store)
    self.assertEqual(len(fig2['data']), 1)
    self.assertEqual(fig2['data'][0]['marker']['size'], 1)
    self.assertEqual(list(fig2['data'][0]['x']), [])
    self.assertEqual(list(fig2['data'][0]['y']), [])
    self.assertEqual(decode_store_data(new_store), {'streams': {id(selection1d): {'index': []}}, 'reset_nclicks': 1, 'kdims': {}})