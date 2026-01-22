import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
class ChainProvider(BaseProvider):
    """This provider wraps one or more other providers.

    Each provider in the chain is called, the first one returning a non-None
    value is then returned.
    """

    def __init__(self, providers=None, conversion_func=None):
        """Initalize a ChainProvider.

        :type providers: list
        :param providers: The initial list of providers to check for values
            when invoked.

        :type conversion_func: None or callable
        :param conversion_func: If this value is None then it has no affect on
            the return type. Otherwise, it is treated as a function that will
            transform provided value.
        """
        if providers is None:
            providers = []
        self._providers = providers
        self._conversion_func = conversion_func

    def __deepcopy__(self, memo):
        return ChainProvider(copy.deepcopy(self._providers, memo), self._conversion_func)

    def provide(self):
        """Provide the value from the first provider to return non-None.

        Each provider in the chain has its provide method called. The first
        one in the chain to return a non-None value is the returned from the
        ChainProvider. When no non-None value is found, None is returned.
        """
        for provider in self._providers:
            value = provider.provide()
            if value is not None:
                return self._convert_type(value)
        return None

    def set_default_provider(self, default_provider):
        if self._providers and isinstance(self._providers[-1], ConstantProvider):
            self._providers[-1] = default_provider
        else:
            self._providers.append(default_provider)
        num_of_constants = sum((isinstance(provider, ConstantProvider) for provider in self._providers))
        if num_of_constants > 1:
            logger.info('ChainProvider object contains multiple instances of ConstantProvider objects')

    def _convert_type(self, value):
        if self._conversion_func is not None:
            return self._conversion_func(value)
        return value

    def __repr__(self):
        return '[%s]' % ', '.join([str(p) for p in self._providers])