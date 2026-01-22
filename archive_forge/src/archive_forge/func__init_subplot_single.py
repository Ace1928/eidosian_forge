import collections
def _init_subplot_single(layout, subplot_type, x_domain, y_domain, max_subplot_ids=None):
    if max_subplot_ids is None:
        max_subplot_ids = _get_initial_max_subplot_ids()
    cnt = max_subplot_ids[subplot_type] + 1
    label = '{subplot_type}{cnt}'.format(subplot_type=subplot_type, cnt=cnt if cnt > 1 else '')
    scene = dict(domain={'x': x_domain, 'y': y_domain})
    layout[label] = scene
    trace_key = 'subplot' if subplot_type in _subplot_prop_named_subplot else subplot_type
    subplot_ref = SubplotRef(subplot_type=subplot_type, layout_keys=(label,), trace_kwargs={trace_key: label})
    max_subplot_ids[subplot_type] = cnt
    return (subplot_ref,)