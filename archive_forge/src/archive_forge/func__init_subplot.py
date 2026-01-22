import collections
def _init_subplot(layout, subplot_type, secondary_y, x_domain, y_domain, max_subplot_ids=None):
    subplot_type = _validate_coerce_subplot_type(subplot_type)
    if max_subplot_ids is None:
        max_subplot_ids = _get_initial_max_subplot_ids()
    x_domain = [max(0.0, x_domain[0]), min(1.0, x_domain[1])]
    y_domain = [max(0.0, y_domain[0]), min(1.0, y_domain[1])]
    if subplot_type == 'xy':
        subplot_refs = _init_subplot_xy(layout, secondary_y, x_domain, y_domain, max_subplot_ids)
    elif subplot_type in _single_subplot_types:
        subplot_refs = _init_subplot_single(layout, subplot_type, x_domain, y_domain, max_subplot_ids)
    elif subplot_type == 'domain':
        subplot_refs = _init_subplot_domain(x_domain, y_domain)
    else:
        raise ValueError('Unsupported subplot type: {}'.format(repr(subplot_type)))
    return subplot_refs