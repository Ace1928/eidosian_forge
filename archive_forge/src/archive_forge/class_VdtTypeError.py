import re
import sys
from pprint import pprint
class VdtTypeError(ValidateError):
    """The value supplied was of the wrong type"""

    def __init__(self, value):
        """
        >>> raise VdtTypeError('jedi')
        Traceback (most recent call last):
        VdtTypeError: the value "jedi" is of the wrong type.
        """
        ValidateError.__init__(self, 'the value "%s" is of the wrong type.' % (value,))