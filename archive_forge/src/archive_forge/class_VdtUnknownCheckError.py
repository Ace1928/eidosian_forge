import re
import sys
from pprint import pprint
class VdtUnknownCheckError(ValidateError):
    """An unknown check function was requested"""

    def __init__(self, value):
        """
        >>> raise VdtUnknownCheckError('yoda')
        Traceback (most recent call last):
        VdtUnknownCheckError: the check "yoda" is unknown.
        """
        ValidateError.__init__(self, 'the check "%s" is unknown.' % (value,))