import re
import sys
from pprint import pprint
class VdtParamError(SyntaxError):
    """An incorrect parameter was passed"""

    def __init__(self, name, value):
        """
        >>> raise VdtParamError('yoda', 'jedi')
        Traceback (most recent call last):
        VdtParamError: passed an incorrect value "jedi" for parameter "yoda".
        """
        SyntaxError.__init__(self, 'passed an incorrect value "%s" for parameter "%s".' % (value, name))