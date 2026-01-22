from collections import deque
import os
import six
from genshi.compat import numeric_types, StringIO, BytesIO
from genshi.core import Attrs, Stream, StreamEventKind, START, TEXT, _ensure
from genshi.input import ParseError
class TemplateError(Exception):
    """Base exception class for errors related to template processing."""

    def __init__(self, message, filename=None, lineno=-1, offset=-1):
        """Create the exception.
        
        :param message: the error message
        :param filename: the filename of the template
        :param lineno: the number of line in the template at which the error
                       occurred
        :param offset: the column number at which the error occurred
        """
        if filename is None:
            filename = '<string>'
        self.msg = message
        if filename != '<string>' or lineno >= 0:
            message = '%s (%s, line %d)' % (self.msg, filename, lineno)
        Exception.__init__(self, message)
        self.filename = filename
        self.lineno = lineno
        self.offset = offset