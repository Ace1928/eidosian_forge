from ._base import *
class LimitExcept(Container):
    """Container for specifying HTTP method restrictions."""

    def __init__(self, value, *args):
        """Initialize."""
        super(LimitExcept, self).__init__(value, *args)
        self.name = 'limit_except'