import copy
import re
import numpy as np
from plotly import colors
from ...core.util import isfinite, max_range
from ..util import color_intervals, process_cmap
def _normalize_subplot_ids(fig):
    """
    Make sure a layout subplot property is initialized for every subplot that
    is referenced by a trace in the figure.

    For example, if a figure contains a `scatterpolar` trace with the `subplot`
    property set to `polar3`, this function will make sure the figure's layout
    has a `polar3` property, and will initialize it to an empty dict if it
    does not

    Note: This function mutates the input figure dict

    Parameters
    ----------
    fig: dict
        A plotly figure dict
    """
    layout = fig.setdefault('layout', {})
    for trace in fig.get('data', None):
        trace_type = trace.get('type', 'scatter')
        subplot_types = _trace_to_subplot.get(trace_type, [])
        for subplot_type in subplot_types:
            subplot_prop_name = _get_subplot_prop_name(subplot_type)
            subplot_val_prefix = _get_subplot_val_prefix(subplot_type)
            subplot_val = trace.get(subplot_prop_name, subplot_val_prefix)
            subplot_number = _get_subplot_number(subplot_val)
            if subplot_number > 1:
                layout_prop_name = subplot_type + str(subplot_number)
            else:
                layout_prop_name = subplot_type
            if layout_prop_name not in layout:
                layout[layout_prop_name] = {}