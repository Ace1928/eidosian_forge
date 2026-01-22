class OneTimeProperty(object):
    """A descriptor to make special properties that become normal attributes."""

    def __init__(self, func):
        """Create a OneTimeProperty instance.

        Parameters
        ----------
          func : method

            The method that will be called the first time to compute a value.
            Afterwards, the method's name will be a standard attribute holding
            the value of this computation.
        """
        self.getter = func
        self.name = func.__name__

    def __get__(self, obj, type=None):
        """Called on attribute access on the class or instance."""
        if obj is None:
            return self.getter
        val = self.getter(obj)
        setattr(obj, self.name, val)
        return val