from collections import deque
import os
import six
from genshi.compat import numeric_types, StringIO, BytesIO
from genshi.core import Attrs, Stream, StreamEventKind, START, TEXT, _ensure
from genshi.input import ParseError
def _apply_directives(stream, directives, ctxt, vars):
    """Apply the given directives to the stream.
    
    :param stream: the stream the directives should be applied to
    :param directives: the list of directives to apply
    :param ctxt: the `Context`
    :param vars: additional variables that should be available when Python
                 code is executed
    :return: the stream with the given directives applied
    """
    if directives:
        stream = directives[0](iter(stream), directives[1:], ctxt, **vars)
    return stream