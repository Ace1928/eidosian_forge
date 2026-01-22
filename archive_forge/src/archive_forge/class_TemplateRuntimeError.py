from collections import deque
import os
import six
from genshi.compat import numeric_types, StringIO, BytesIO
from genshi.core import Attrs, Stream, StreamEventKind, START, TEXT, _ensure
from genshi.input import ParseError
class TemplateRuntimeError(TemplateError):
    """Exception raised when an the evaluation of a Python expression in a
    template causes an error.
    """