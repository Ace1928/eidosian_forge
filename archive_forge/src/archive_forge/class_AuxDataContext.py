from __future__ import unicode_literals, with_statement
import re
import pybtex.io
from pybtex.errors import report_error
from pybtex.exceptions import PybtexError
from pybtex import py3compat
class AuxDataContext(object):
    lineno = None
    line = None
    filename = None

    def __init__(self, filename):
        self.filename = filename