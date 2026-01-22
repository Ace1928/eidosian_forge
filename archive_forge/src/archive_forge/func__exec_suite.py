from collections import deque
import os
import six
from genshi.compat import numeric_types, StringIO, BytesIO
from genshi.core import Attrs, Stream, StreamEventKind, START, TEXT, _ensure
from genshi.input import ParseError
def _exec_suite(suite, ctxt, vars=None):
    """Execute the given `Suite` object.
    
    :param suite: the code suite to execute
    :param ctxt: the `Context`
    :param vars: additional variables that should be available to the
                 code
    """
    if vars:
        ctxt.push(vars)
        ctxt.push({})
    suite.execute(ctxt)
    if vars:
        top = ctxt.pop()
        ctxt.pop()
        ctxt.frames[0].update(top)