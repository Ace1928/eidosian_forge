from mako import util
class CacheImpl:
    """Provide a cache implementation for use by :class:`.Cache`."""

    def __init__(self, cache):
        self.cache = cache
    pass_context = False
    "If ``True``, the :class:`.Context` will be passed to\n    :meth:`get_or_create <.CacheImpl.get_or_create>` as the name ``'context'``.\n    "

    def get_or_create(self, key, creation_function, **kw):
        """Retrieve a value from the cache, using the given creation function
        to generate a new value.

        This function *must* return a value, either from
        the cache, or via the given creation function.
        If the creation function is called, the newly
        created value should be populated into the cache
        under the given key before being returned.

        :param key: the value's key.
        :param creation_function: function that when called generates
         a new value.
        :param \\**kw: cache configuration arguments.

        """
        raise NotImplementedError()

    def set(self, key, value, **kw):
        """Place a value in the cache.

        :param key: the value's key.
        :param value: the value.
        :param \\**kw: cache configuration arguments.

        """
        raise NotImplementedError()

    def get(self, key, **kw):
        """Retrieve a value from the cache.

        :param key: the value's key.
        :param \\**kw: cache configuration arguments.

        """
        raise NotImplementedError()

    def invalidate(self, key, **kw):
        """Invalidate a value in the cache.

        :param key: the value's key.
        :param \\**kw: cache configuration arguments.

        """
        raise NotImplementedError()