from collections import deque
import os
import six
from genshi.compat import numeric_types, StringIO, BytesIO
from genshi.core import Attrs, Stream, StreamEventKind, START, TEXT, _ensure
from genshi.input import ParseError
def _eval_expr(expr, ctxt, vars=None):
    """Evaluate the given `Expression` object.
    
    :param expr: the expression to evaluate
    :param ctxt: the `Context`
    :param vars: additional variables that should be available to the
                 expression
    :return: the result of the evaluation
    """
    if vars:
        ctxt.push(vars)
    retval = expr.evaluate(ctxt)
    if vars:
        ctxt.pop()
    return retval