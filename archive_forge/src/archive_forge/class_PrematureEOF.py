from __future__ import unicode_literals
import re
from pybtex.exceptions import PybtexError
from pybtex import py3compat
class PrematureEOF(PybtexSyntaxError):

    def __init__(self, parser):
        message = 'premature end of file'
        super(PrematureEOF, self).__init__(message, parser)