import re
import sys
from pprint import pprint
class VdtValueTooSmallError(VdtValueError):
    """The value supplied was of the correct type, but was too small."""

    def __init__(self, value):
        """
        >>> raise VdtValueTooSmallError('0')
        Traceback (most recent call last):
        VdtValueTooSmallError: the value "0" is too small.
        """
        ValidateError.__init__(self, 'the value "%s" is too small.' % (value,))