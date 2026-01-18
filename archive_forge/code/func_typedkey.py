def typedkey(*args, **kwargs):
    """Return a typed cache key for the specified hashable arguments."""
    key = hashkey(*args, **kwargs)
    key += tuple((type(v) for v in args))
    key += tuple((type(v) for _, v in sorted(kwargs.items())))
    return key