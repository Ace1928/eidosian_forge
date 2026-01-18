import base64
import copy
import pickle
import uuid
from collections import namedtuple
from dash.exceptions import PreventUpdate
import holoviews as hv
from holoviews.core.decollate import (
from holoviews.plotting.plotly import DynamicMap, PlotlyRenderer
from holoviews.plotting.plotly.callbacks import (
from holoviews.plotting.plotly.util import clean_internal_figure_properties
from holoviews.streams import Derived, History
import plotly.graph_objects as go
from dash import callback_context
from dash.dependencies import Input, Output, State
@app.callback(outputs, inputs, states)
def update_figure(*args):
    triggered_prop_ids = {entry['prop_id'] for entry in callback_context.triggered}
    selected_dicts = [args[j] or {} for j in range(0, num_figs * 2, 2)]
    relayout_dicts = [args[j] or {} for j in range(1, num_figs * 2, 2)]
    any_change = False
    store_data = decode_store_data(args[-1])
    reset_nclicks = 0
    if reset_button:
        reset_nclicks = args[-2] or 0
        prior_reset_nclicks = store_data.get('reset_nclicks', 0)
        if reset_nclicks != prior_reset_nclicks:
            store_data['reset_nclicks'] = reset_nclicks
            store_data['streams'] = copy.deepcopy(initial_stream_contents)
            selected_dicts = [None for _ in selected_dicts]
            relayout_dicts = [None for _ in relayout_dicts]
            any_change = True
    if store_data is None:
        store_data = {'streams': {}}
    store_data.setdefault('kdims', {})
    for i, kdim in zip(range(num_figs * 2, num_figs * 2 + len(all_kdims)), all_kdims):
        if kdim not in store_data['kdims'] or store_data['kdims'][kdim] != args[i]:
            store_data['kdims'][kdim] = args[i]
            any_change = True
    for fig_ind in range(len(initial_fig_dicts)):
        graph_id = graph_ids[fig_ind]
        for plotly_stream_type, uid_to_streams_for_type in uid_to_stream_ids.items():
            for panel_prop in plotly_stream_type.callback_properties:
                if panel_prop == 'selected_data':
                    if graph_id + '.selectedData' in triggered_prop_ids:
                        stream_event_data = plotly_stream_type.get_event_data_from_property_update(panel_prop, selected_dicts[fig_ind], initial_fig_dicts[fig_ind])
                        any_change = update_stream_values_for_type(store_data, stream_event_data, uid_to_streams_for_type) or any_change
                elif panel_prop == 'viewport':
                    if graph_id + '.relayoutData' in triggered_prop_ids:
                        stream_event_data = plotly_stream_type.get_event_data_from_property_update(panel_prop, relayout_dicts[fig_ind], initial_fig_dicts[fig_ind])
                        stream_event_data = {uid: event_data for uid, event_data in stream_event_data.items() if event_data['x_range'] is not None or event_data['y_range'] is not None}
                        any_change = update_stream_values_for_type(store_data, stream_event_data, uid_to_streams_for_type) or any_change
                elif panel_prop == 'relayout_data':
                    if graph_id + '.relayoutData' in triggered_prop_ids:
                        stream_event_data = plotly_stream_type.get_event_data_from_property_update(panel_prop, relayout_dicts[fig_ind], initial_fig_dicts[fig_ind])
                        any_change = update_stream_values_for_type(store_data, stream_event_data, uid_to_streams_for_type) or any_change
    if not any_change:
        raise PreventUpdate
    for output_id in reversed(stream_callbacks):
        stream_callback = stream_callbacks[output_id]
        input_ids = stream_callback.input_ids
        fn = stream_callback.fn
        output_id = stream_callback.output_id
        input_values = [store_data['streams'][input_id] for input_id in input_ids]
        output_value = fn(*input_values)
        store_data['streams'][output_id] = output_value
    figs = [None] * num_figs
    for fig_ind, (fn, stream_ids) in fig_to_fn_stream_ids.items():
        fig_kdim_values = [store_data['kdims'][kd] for kd in kdims_per_fig[fig_ind]]
        stream_values = [store_data['streams'][stream_id] for stream_id in stream_ids]
        hvobj = fn(*fig_kdim_values + stream_values)
        plot = PlotlyRenderer.get_plot(hvobj)
        fig = plot_to_figure(plot, reset_nclicks=reset_nclicks, layout_ranges=layout_ranges[fig_ind], responsive=responsive, use_ranges=use_ranges).to_dict()
        figs[fig_ind] = fig
    return figs + [encode_store_data(store_data)]