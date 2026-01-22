def annotation_params_for_rect(shape_type, shape_args, position):
    x0 = shape_args['x0']
    x1 = shape_args['x1']
    y0 = shape_args['y0']
    y1 = shape_args['y1']
    position, pos_str = _prepare_position(position, prepend_inside=True)
    if position == set(['inside', 'top', 'left']):
        return _df_anno('left', 'top', min([x0, x1]), max([y0, y1]))
    if position == set(['inside', 'top', 'right']):
        return _df_anno('right', 'top', max([x0, x1]), max([y0, y1]))
    if position == set(['inside', 'top']):
        return _df_anno('center', 'top', _mean([x0, x1]), max([y0, y1]))
    if position == set(['inside', 'bottom', 'left']):
        return _df_anno('left', 'bottom', min([x0, x1]), min([y0, y1]))
    if position == set(['inside', 'bottom', 'right']):
        return _df_anno('right', 'bottom', max([x0, x1]), min([y0, y1]))
    if position == set(['inside', 'bottom']):
        return _df_anno('center', 'bottom', _mean([x0, x1]), min([y0, y1]))
    if position == set(['inside', 'left']):
        return _df_anno('left', 'middle', min([x0, x1]), _mean([y0, y1]))
    if position == set(['inside', 'right']):
        return _df_anno('right', 'middle', max([x0, x1]), _mean([y0, y1]))
    if position == set(['inside']):
        return _df_anno('center', 'middle', _mean([x0, x1]), _mean([y0, y1]))
    if position == set(['outside', 'top', 'left']):
        return _df_anno('right' if shape_type == 'vrect' else 'left', 'bottom' if shape_type == 'hrect' else 'top', min([x0, x1]), max([y0, y1]))
    if position == set(['outside', 'top', 'right']):
        return _df_anno('left' if shape_type == 'vrect' else 'right', 'bottom' if shape_type == 'hrect' else 'top', max([x0, x1]), max([y0, y1]))
    if position == set(['outside', 'top']):
        return _df_anno('center', 'bottom', _mean([x0, x1]), max([y0, y1]))
    if position == set(['outside', 'bottom', 'left']):
        return _df_anno('right' if shape_type == 'vrect' else 'left', 'top' if shape_type == 'hrect' else 'bottom', min([x0, x1]), min([y0, y1]))
    if position == set(['outside', 'bottom', 'right']):
        return _df_anno('left' if shape_type == 'vrect' else 'right', 'top' if shape_type == 'hrect' else 'bottom', max([x0, x1]), min([y0, y1]))
    if position == set(['outside', 'bottom']):
        return _df_anno('center', 'top', _mean([x0, x1]), min([y0, y1]))
    if position == set(['outside', 'left']):
        return _df_anno('right', 'middle', min([x0, x1]), _mean([y0, y1]))
    if position == set(['outside', 'right']):
        return _df_anno('left', 'middle', max([x0, x1]), _mean([y0, y1]))
    raise ValueError('Invalid annotation position %s' % (pos_str,))