import warnings
def _prepare_initializer(initializer, initargs):
    if initializer is not None and (not callable(initializer)):
        raise TypeError(f'initializer must be a callable, got: {initializer!r}')
    return _chain_initializers([(initializer, initargs), _make_viztracer_initializer_and_initargs()])