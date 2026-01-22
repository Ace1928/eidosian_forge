import copy
import re
import numpy as np
from plotly import colors
from ...core.util import isfinite, max_range
from ..util import color_intervals, process_cmap
def _offset_subplot_ids(fig, offsets):
    """
    Apply offsets to the subplot id numbers in a figure.

    Note: This function mutates the input figure dict

    Note: This function assumes that the normalize_subplot_ids function has
    already been run on the figure, so that all layout subplot properties in
    use are explicitly present in the figure's layout.

    Parameters
    ----------
    fig: dict
        A plotly figure dict
    offsets: dict
        A dict from subplot types to the offset to be applied for each subplot
        type.  This dict matches the form of the dict returned by
        get_max_subplot_ids
    """
    for trace in fig.get('data', None):
        trace_type = trace.get('type', 'scatter')
        subplot_types = _trace_to_subplot.get(trace_type, [])
        for subplot_type in subplot_types:
            subplot_prop_name = _get_subplot_prop_name(subplot_type)
            subplot_val_prefix = _get_subplot_val_prefix(subplot_type)
            subplot_val = trace.get(subplot_prop_name, subplot_val_prefix)
            subplot_number = _get_subplot_number(subplot_val)
            offset_subplot_number = subplot_number + offsets.get(subplot_type, 0)
            if offset_subplot_number > 1:
                trace[subplot_prop_name] = subplot_val_prefix + str(offset_subplot_number)
            else:
                trace[subplot_prop_name] = subplot_val_prefix
    layout = fig.setdefault('layout', {})
    new_subplots = {}
    for subplot_type in offsets:
        offset = offsets[subplot_type]
        if offset < 1:
            continue
        for layout_prop in list(layout.keys()):
            if layout_prop.startswith(subplot_type):
                subplot_number = _get_subplot_number(layout_prop)
                new_subplot_number = subplot_number + offset
                new_layout_prop = subplot_type + str(new_subplot_number)
                new_subplots[new_layout_prop] = layout.pop(layout_prop)
    layout.update(new_subplots)
    x_offset = offsets.get('xaxis', 0)
    y_offset = offsets.get('yaxis', 0)
    for layout_prop in list(layout.keys()):
        if layout_prop.startswith('xaxis'):
            xaxis = layout[layout_prop]
            anchor = xaxis.get('anchor', 'y')
            anchor_number = _get_subplot_number(anchor) + y_offset
            if anchor_number > 1:
                xaxis['anchor'] = 'y' + str(anchor_number)
            else:
                xaxis['anchor'] = 'y'
        elif layout_prop.startswith('yaxis'):
            yaxis = layout[layout_prop]
            anchor = yaxis.get('anchor', 'x')
            anchor_number = _get_subplot_number(anchor) + x_offset
            if anchor_number > 1:
                yaxis['anchor'] = 'x' + str(anchor_number)
            else:
                yaxis['anchor'] = 'x'
    for layout_prop in list(layout.keys()):
        if layout_prop[1:5] == 'axis':
            axis = layout[layout_prop]
            matches_val = axis.get('matches', None)
            if matches_val:
                if matches_val[0] == 'x':
                    matches_number = _get_subplot_number(matches_val) + x_offset
                elif matches_val[0] == 'y':
                    matches_number = _get_subplot_number(matches_val) + y_offset
                else:
                    continue
                suffix = str(matches_number) if matches_number > 1 else ''
                axis['matches'] = matches_val[0] + suffix
    for layout_prop in ['annotations', 'shapes', 'images']:
        for obj in layout.get(layout_prop, []):
            if x_offset:
                xref = obj.get('xref', 'x')
                if xref != 'paper':
                    xref_number = _get_subplot_number(xref)
                    obj['xref'] = 'x' + str(xref_number + x_offset)
            if y_offset:
                yref = obj.get('yref', 'y')
                if yref != 'paper':
                    yref_number = _get_subplot_number(yref)
                    obj['yref'] = 'y' + str(yref_number + y_offset)