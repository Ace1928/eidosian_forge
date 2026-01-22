import warnings
class _ChainedInitializer:
    """Compound worker initializer

    This is meant to be used in conjunction with _chain_initializers to
    produce  the necessary chained_args list to be passed to __call__.
    """

    def __init__(self, initializers):
        self._initializers = initializers

    def __call__(self, *chained_args):
        for initializer, args in zip(self._initializers, chained_args):
            initializer(*args)