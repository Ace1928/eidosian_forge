import collections
def update_axis_matches(first_axis_id, subplot_ref, spec, remove_label):
    if subplot_ref is None:
        return first_axis_id
    if x_or_y == 'x':
        span = spec['colspan']
    else:
        span = spec['rowspan']
    if subplot_ref.subplot_type == 'xy' and span == 1:
        if first_axis_id is None:
            first_axis_name = subplot_ref.layout_keys[layout_key_ind]
            first_axis_id = first_axis_name.replace('axis', '')
        else:
            axis_name = subplot_ref.layout_keys[layout_key_ind]
            axis_to_match = layout[axis_name]
            axis_to_match.matches = first_axis_id
            if remove_label:
                axis_to_match.showticklabels = False
    return first_axis_id