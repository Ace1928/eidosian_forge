from collections import deque
import os
import six
from genshi.compat import numeric_types, StringIO, BytesIO
from genshi.core import Attrs, Stream, StreamEventKind, START, TEXT, _ensure
from genshi.input import ParseError
class TemplateSyntaxError(TemplateError):
    """Exception raised when an expression in a template causes a Python syntax
    error, or the template is not well-formed.
    """

    def __init__(self, message, filename=None, lineno=-1, offset=-1):
        """Create the exception
        
        :param message: the error message
        :param filename: the filename of the template
        :param lineno: the number of line in the template at which the error
                       occurred
        :param offset: the column number at which the error occurred
        """
        if isinstance(message, SyntaxError) and message.lineno is not None:
            message = str(message).replace(' (line %d)' % message.lineno, '')
        TemplateError.__init__(self, message, filename, lineno)