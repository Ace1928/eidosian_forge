import textwrap
import pkgutil
import copy
import os
import json
from functools import reduce
def to_templated(fig, skip=('title', 'text')):
    """
    Return a copy of a figure where all styling properties have been moved
    into the figure's template.  The template property of the resulting figure
    may then be used to set the default styling of other figures.

    Parameters
    ----------
    fig
        Figure object or dict representing a figure
    skip
        A collection of names of properties to skip when moving properties to
        the template. Defaults to ('title', 'text') so that the text
        of figure titles, axis titles, and annotations does not become part of
        the template

    Examples
    --------
    Imports

    >>> import plotly.graph_objs as go
    >>> import plotly.io as pio

    Construct a figure with large courier text

    >>> fig = go.Figure(layout={'title': 'Figure Title',
    ...                         'font': {'size': 20, 'family': 'Courier'},
    ...                         'template':"none"})
    >>> fig # doctest: +NORMALIZE_WHITESPACE
    Figure({
        'data': [],
        'layout': {'font': {'family': 'Courier', 'size': 20},
                   'template': '...', 'title': {'text': 'Figure Title'}}
    })

    Convert to a figure with a template. Note how the 'font' properties have
    been moved into the template property.

    >>> templated_fig = pio.to_templated(fig)
    >>> templated_fig.layout.template
    layout.Template({
        'layout': {'font': {'family': 'Courier', 'size': 20}}
    })
    >>> templated_fig
    Figure({
        'data': [], 'layout': {'template': '...', 'title': {'text': 'Figure Title'}}
    })


    Next create a new figure with this template

    >>> fig2 = go.Figure(layout={
    ...     'title': 'Figure 2 Title',
    ...     'template': templated_fig.layout.template})
    >>> fig2.layout.template
    layout.Template({
        'layout': {'font': {'family': 'Courier', 'size': 20}}
    })

    The default font in fig2 will now be size 20 Courier.

    Next, register as a named template...

    >>> pio.templates['large_courier'] = templated_fig.layout.template

    and specify this template by name when constructing a figure.

    >>> go.Figure(layout={
    ...     'title': 'Figure 3 Title',
    ...     'template': 'large_courier'}) # doctest: +ELLIPSIS
    Figure(...)

    Finally, set this as the default template to be applied to all new figures

    >>> pio.templates.default = 'large_courier'
    >>> fig = go.Figure(layout={'title': 'Figure 4 Title'})
    >>> fig.layout.template
    layout.Template({
        'layout': {'font': {'family': 'Courier', 'size': 20}}
    })

    Returns
    -------
    go.Figure
    """
    from plotly.basedatatypes import BaseFigure
    from plotly.graph_objs import Figure
    if not isinstance(fig, BaseFigure):
        fig = Figure(fig)
    if not skip:
        skip = set()
    else:
        skip = set(skip)
    skip.add('uid')
    templated_fig = copy.deepcopy(fig)
    walk_push_to_template(templated_fig.layout, templated_fig.layout.template.layout, skip=skip)
    trace_type_indexes = {}
    for trace in list(templated_fig.data):
        template_index = trace_type_indexes.get(trace.type, 0)
        template_traces = list(templated_fig.layout.template.data[trace.type])
        while len(template_traces) <= template_index:
            template_traces.append(trace.__class__())
        template_trace = template_traces[template_index]
        walk_push_to_template(trace, template_trace, skip=skip)
        templated_fig.layout.template.data[trace.type] = template_traces
        trace_type_indexes[trace.type] = template_index + 1
    any_non_empty = False
    for trace_type in templated_fig.layout.template.data:
        traces = templated_fig.layout.template.data[trace_type]
        is_empty = [trace.to_plotly_json() == {'type': trace_type} for trace in traces]
        if all(is_empty):
            templated_fig.layout.template.data[trace_type] = None
        else:
            any_non_empty = True
    if not any_non_empty:
        templated_fig.layout.template.data = None
    return templated_fig