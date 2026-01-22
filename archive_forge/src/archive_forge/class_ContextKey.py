class ContextKey(object):
    """Provides a unique key suitable for use as a key into AuthContext."""

    def __init__(self, name):
        """Creates a context key using the given name. The name is
     only for informational purposes.
     """
        self._name = name

    def __str__(self):
        return '%s#%#x' % (self._name, id(self))

    def __repr__(self):
        return 'context_key(%r, %#x)' % (self._name, id(self))