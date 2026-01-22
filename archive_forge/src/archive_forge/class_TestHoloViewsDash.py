import unittest
from unittest.mock import MagicMock, patch
from dash._callback_context import CallbackContext
from holoviews import Bounds, DynamicMap, Scatter
from holoviews.plotting.plotly.dash import (
from holoviews.streams import BoundsXY, RangeXY, Selection1D
from .test_plot import TestPlotlyPlot
import plotly.io as pio
class TestHoloViewsDash(TestPlotlyPlot):

    def setUp(self):
        super().setUp()
        self.app = MagicMock()
        self.decorator = MagicMock()
        self.app.callback.return_value = self.decorator

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

    def test_boundsxy_dynamic_map(self):
        scatter = Scatter([0, 0])
        boundsxy = BoundsXY(source=scatter)
        dmap = DynamicMap(lambda bounds: Bounds(bounds) if bounds is not None else Bounds((0, 0, 0, 0)), streams=[boundsxy])
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
        self.assertEqual(fig1['data'][0]['type'], 'scatter')
        self.assertEqual(len(fig2['data']), 0)
        self.assertEqual(len(fig2['layout']['shapes']), 1)
        self.assertEqual(fig2['layout']['shapes'][0]['path'], 'M0 0L0 0L0 0L0 0L0 0Z')
        callback_fn = self.app.callback.return_value.call_args[0][0]
        store_value = encode_store_data({'streams': {id(boundsxy): boundsxy.contents}})
        with patch.object(CallbackContext, 'triggered', [{'prop_id': inputs[0].component_id + '.selectedData'}]):
            [fig1, fig2, new_store] = callback_fn({'range': {'x': [1, 2], 'y': [3, 4]}}, {}, {}, {}, 0, store_value)
        self.assertEqual(fig1['data'][0]['type'], 'scatter')
        self.assertEqual(len(fig2['data']), 0)
        self.assertEqual(len(fig2['layout']['shapes']), 1)
        self.assertEqual(fig2['layout']['shapes'][0]['path'], 'M1 3L1 4L2 4L2 3L1 3Z')
        self.assertEqual(decode_store_data(new_store), {'streams': {id(boundsxy): {'bounds': (1, 3, 2, 4)}}, 'kdims': {}})
        with patch.object(CallbackContext, 'triggered', [{'prop_id': components.resets[0].id + '.n_clicks'}]):
            [fig1, fig2, new_store] = callback_fn({'range': {'x': [1, 2], 'y': [3, 4]}}, {}, {}, {}, 1, store_value)
        self.assertEqual(fig1['data'][0]['type'], 'scatter')
        self.assertEqual(len(fig2['data']), 0)
        self.assertEqual(len(fig2['layout']['shapes']), 1)
        self.assertEqual(fig2['layout']['shapes'][0]['path'], 'M0 0L0 0L0 0L0 0L0 0Z')
        self.assertEqual(decode_store_data(new_store), {'streams': {id(boundsxy): {'bounds': None}}, 'reset_nclicks': 1, 'kdims': {}})

    def test_rangexy_dynamic_map(self):
        scatter = Scatter([[0, 1], [0, 1]], kdims=['x'], vdims=['y'])
        rangexy = RangeXY(source=scatter)

        def dmap_fn(x_range, y_range):
            x_range = (0, 1) if x_range is None else x_range
            y_range = (0, 1) if y_range is None else y_range
            return Scatter([[x_range[0], y_range[0]], [x_range[1], y_range[1]]], kdims=['x1'], vdims=['y1'])
        dmap = DynamicMap(dmap_fn, streams=[rangexy])
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
        callback_fn = self.app.callback.return_value.call_args[0][0]
        store_value = encode_store_data({'streams': {id(rangexy): rangexy.contents}})
        with patch.object(CallbackContext, 'triggered', [{'prop_id': components.graphs[0].id + '.relayoutData'}]):
            [fig1, fig2, new_store] = callback_fn({}, {'xaxis.range[0]': 1, 'xaxis.range[1]': 3, 'yaxis.range[0]': 2, 'yaxis.range[1]': 4}, {}, {}, None, store_value)
        self.assertEqual(fig1['data'][0]['type'], 'scatter')
        self.assertEqual(len(fig2['data']), 1)
        self.assertEqual(list(fig2['data'][0]['x']), [1, 3])
        self.assertEqual(list(fig2['data'][0]['y']), [2, 4])
        self.assertEqual(decode_store_data(new_store), {'streams': {id(rangexy): {'x_range': (1, 3), 'y_range': (2, 4)}}, 'kdims': {}})

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

    def test_kdims_dynamic_map(self):
        dmap = DynamicMap(lambda kdim1: Scatter([kdim1, kdim1]), kdims=['kdim1']).redim.values(kdim1=[1, 2, 3, 4])
        components = to_dash(self.app, [dmap])
        self.assertIsInstance(components, DashComponents)
        self.assertEqual(len(components.graphs), 1)
        self.assertEqual(len(components.kdims), 1)
        self.assertIsInstance(components.store, Store)
        self.assertEqual(len(components.resets), 0)
        decorator_args = next(iter(self.app.callback.call_args_list[0]))
        outputs, inputs, states = decorator_args
        expected_outputs = [(g.id, 'figure') for g in components.graphs] + [(components.store.id, 'data')]
        self.assertEqual([(output.component_id, output.component_property) for output in outputs], expected_outputs)
        expected_inputs = [(g.id, prop) for g in components.graphs for prop in ['selectedData', 'relayoutData']] + [(next(iter(components.kdims.values())).children[1].id, 'value')]
        self.assertEqual([(ip.component_id, ip.component_property) for ip in inputs], expected_inputs)
        expected_state = [(components.store.id, 'data')]
        self.assertEqual([(state.component_id, state.component_property) for state in states], expected_state)
        callback_fn = self.decorator.call_args_list[0][0][0]
        store_value = encode_store_data({'streams': {}})
        with patch.object(CallbackContext, 'triggered', []):
            [fig, new_store] = callback_fn({}, {}, 3, None, store_value)
        self.assertEqual(fig['data'][0]['type'], 'scatter')
        self.assertEqual(list(fig['data'][0]['x']), [0, 1])
        self.assertEqual(list(fig['data'][0]['y']), [3, 3])