import collections
def _validate_coerce_subplot_type(subplot_type):
    orig_subplot_type = subplot_type
    subplot_type = subplot_type.lower()
    if subplot_type in _subplot_types:
        return subplot_type
    subplot_type = _subplot_type_for_trace_type(subplot_type)
    if subplot_type is None:
        raise ValueError('Unsupported subplot type: {}'.format(repr(orig_subplot_type)))
    else:
        return subplot_type