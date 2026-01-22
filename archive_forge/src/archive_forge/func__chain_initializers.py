import warnings
def _chain_initializers(initializer_and_args):
    """Convenience helper to combine a sequence of initializers.

    If some initializers are None, they are filtered out.
    """
    filtered_initializers = []
    filtered_initargs = []
    for initializer, initargs in initializer_and_args:
        if initializer is not None:
            filtered_initializers.append(initializer)
            filtered_initargs.append(initargs)
    if not filtered_initializers:
        return (None, ())
    elif len(filtered_initializers) == 1:
        return (filtered_initializers[0], filtered_initargs[0])
    else:
        return (_ChainedInitializer(filtered_initializers), filtered_initargs)