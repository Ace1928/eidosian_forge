def _add_inside_to_position(pos):
    if not ('inside' in pos or 'outside' in pos):
        pos.add('inside')
    return pos