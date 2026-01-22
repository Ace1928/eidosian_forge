from __future__ import unicode_literals, with_statement
import re
import pybtex.io
from pybtex.errors import report_error
from pybtex.exceptions import PybtexError
from pybtex import py3compat
@py3compat.python_2_unicode_compatible
class AuxDataError(PybtexError):

    def __init__(self, message, context=None):
        super(AuxDataError, self).__init__(message, context.filename)
        self.context = context

    def get_context(self):
        if self.context.line:
            marker = '^' * len(self.context.line)
            return self.context.line + '\n' + marker

    def __str__(self):
        base_message = py3compat.__str__(super(AuxDataError, self))
        lineno = self.context.lineno
        location = 'in line {0}: '.format(lineno) if lineno else ''
        return location + base_message