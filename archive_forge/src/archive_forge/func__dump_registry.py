def _dump_registry(cls, file=None):
    """Debug helper to print the ABC registry."""
    print(f'Class: {cls.__module__}.{cls.__qualname__}', file=file)
    print(f'Inv. counter: {get_cache_token()}', file=file)
    _abc_registry, _abc_cache, _abc_negative_cache, _abc_negative_cache_version = _get_dump(cls)
    print(f'_abc_registry: {_abc_registry!r}', file=file)
    print(f'_abc_cache: {_abc_cache!r}', file=file)
    print(f'_abc_negative_cache: {_abc_negative_cache!r}', file=file)
    print(f'_abc_negative_cache_version: {_abc_negative_cache_version!r}', file=file)