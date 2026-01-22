def _by_class(*args, **kw):
    cls = args[0].__class__
    for t in type(cls.__name__, (cls, object), {}).__mro__:
        f = _gbt(t, _sentinel)
        if f is not _sentinel:
            return f(*args, **kw)
    else:
        return func(*args, **kw)