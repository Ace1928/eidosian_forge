import copy
import re
import numpy as np
from plotly import colors
from ...core.util import isfinite, max_range
from ..util import color_intervals, process_cmap
def _get_max_subplot_ids(fig):
    """
    Given an input figure, return a dict containing the max subplot number
    for each subplot type in the figure

    Parameters
    ----------
    fig: dict
        A plotly figure dict

    Returns
    -------
    dict
        A dict from subplot type strings to integers indicating the largest
        subplot number in the figure of that subplot type
    """
    max_subplot_ids = {subplot_type: 0 for subplot_type in _subplot_types}
    max_subplot_ids['xaxis'] = 0
    max_subplot_ids['yaxis'] = 0
    for trace in fig.get('data', []):
        trace_type = trace.get('type', 'scatter')
        subplot_types = _trace_to_subplot.get(trace_type, [])
        for subplot_type in subplot_types:
            subplot_prop_name = _get_subplot_prop_name(subplot_type)
            subplot_val_prefix = _get_subplot_val_prefix(subplot_type)
            subplot_val = trace.get(subplot_prop_name, subplot_val_prefix)
            subplot_number = _get_subplot_number(subplot_val)
            max_subplot_ids[subplot_type] = max(max_subplot_ids[subplot_type], subplot_number)
    layout = fig.get('layout', {})
    for layout_prop in ['annotations', 'shapes', 'images']:
        for obj in layout.get(layout_prop, []):
            xref = obj.get('xref', 'x')
            if xref != 'paper':
                xref_number = _get_subplot_number(xref)
                max_subplot_ids['xaxis'] = max(max_subplot_ids['xaxis'], xref_number)
            yref = obj.get('yref', 'y')
            if yref != 'paper':
                yref_number = _get_subplot_number(yref)
                max_subplot_ids['yaxis'] = max(max_subplot_ids['yaxis'], yref_number)
    return max_subplot_ids