from functools import total_ordering
from ._funcs import astuple
from ._make import attrib, attrs

        Ensure *other* is a tuple of a valid length.

        Returns a possibly transformed *other* and ourselves as a tuple of
        the same length as *other*.
        