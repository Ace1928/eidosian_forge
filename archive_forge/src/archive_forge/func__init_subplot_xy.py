import collections
def _init_subplot_xy(layout, secondary_y, x_domain, y_domain, max_subplot_ids=None):
    if max_subplot_ids is None:
        max_subplot_ids = _get_initial_max_subplot_ids()
    x_cnt = max_subplot_ids['xaxis'] + 1
    y_cnt = max_subplot_ids['yaxis'] + 1
    x_label = 'x{cnt}'.format(cnt=x_cnt if x_cnt > 1 else '')
    y_label = 'y{cnt}'.format(cnt=y_cnt if y_cnt > 1 else '')
    x_anchor, y_anchor = (y_label, x_label)
    xaxis_name = 'xaxis{cnt}'.format(cnt=x_cnt if x_cnt > 1 else '')
    yaxis_name = 'yaxis{cnt}'.format(cnt=y_cnt if y_cnt > 1 else '')
    x_axis = {'domain': x_domain, 'anchor': x_anchor}
    y_axis = {'domain': y_domain, 'anchor': y_anchor}
    layout[xaxis_name] = x_axis
    layout[yaxis_name] = y_axis
    subplot_refs = [SubplotRef(subplot_type='xy', layout_keys=(xaxis_name, yaxis_name), trace_kwargs={'xaxis': x_label, 'yaxis': y_label})]
    if secondary_y:
        y_cnt += 1
        secondary_yaxis_name = 'yaxis{cnt}'.format(cnt=y_cnt if y_cnt > 1 else '')
        secondary_y_label = 'y{cnt}'.format(cnt=y_cnt)
        subplot_refs.append(SubplotRef(subplot_type='xy', layout_keys=(xaxis_name, secondary_yaxis_name), trace_kwargs={'xaxis': x_label, 'yaxis': secondary_y_label}))
        secondary_y_axis = {'anchor': y_anchor, 'overlaying': y_label, 'side': 'right'}
        layout[secondary_yaxis_name] = secondary_y_axis
    max_subplot_ids['xaxis'] = x_cnt
    max_subplot_ids['yaxis'] = y_cnt
    return tuple(subplot_refs)