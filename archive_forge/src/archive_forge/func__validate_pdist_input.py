import cupy
def _validate_pdist_input(X, m, n, metric_info, **kwargs):
    types = metric_info.types
    typ = types[types.index(X.dtype)] if X.dtype in types else types[0]
    X = _convert_to_type(X, out_type=typ)
    _validate_kwargs = metric_info.validator
    if _validate_kwargs:
        kwargs = _validate_kwargs(X, m, n, **kwargs)
    return (X, typ, kwargs)