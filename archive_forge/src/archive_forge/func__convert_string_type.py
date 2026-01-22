from .util import FileWrapper, guess_scheme, is_hop_by_hop
from .headers import Headers
import sys, os, time
def _convert_string_type(self, value, title):
    """Convert/check value type."""
    if type(value) is str:
        return value
    raise AssertionError('{0} must be of type str (got {1})'.format(title, repr(value)))