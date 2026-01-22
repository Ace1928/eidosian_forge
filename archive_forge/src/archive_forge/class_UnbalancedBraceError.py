from __future__ import unicode_literals
import re
from pybtex.bibtex.utils import bibtex_abbreviate, bibtex_len
from pybtex.database import Person
from pybtex.scanner import (
class UnbalancedBraceError(PybtexSyntaxError):

    def __init__(self, parser):
        message = u'name format string "{0}" has unbalanced braces'.format(parser.text)
        super(UnbalancedBraceError, self).__init__(message, parser)