import re
import sys
from pprint import pprint
class VdtValueTooShortError(VdtValueError):
    """The value supplied was of the correct type, but was too short."""

    def __init__(self, value):
        """
        >>> raise VdtValueTooShortError('jed')
        Traceback (most recent call last):
        VdtValueTooShortError: the value "jed" is too short.
        """
        ValidateError.__init__(self, 'the value "%s" is too short.' % (value,))