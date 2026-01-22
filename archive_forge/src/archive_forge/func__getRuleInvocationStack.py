from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import sys
from antlr3 import runtime_version, runtime_version_str
from antlr3.compat import set, frozenset, reversed
from antlr3.constants import DEFAULT_CHANNEL, HIDDEN_CHANNEL, EOF, \
from antlr3.exceptions import RecognitionException, MismatchedTokenException, \
from antlr3.tokens import CommonToken, EOF_TOKEN, SKIP_TOKEN
import six
from six import unichr
def _getRuleInvocationStack(cls, module):
    """
        A more general version of getRuleInvocationStack where you can
        pass in, for example, a RecognitionException to get it's rule
        stack trace.  This routine is shared with all recognizers, hence,
        static.

        TODO: move to a utility class or something; weird having lexer call
        this
        """
    rules = []
    for frame in reversed(inspect.stack()):
        code = frame[0].f_code
        codeMod = inspect.getmodule(code)
        if codeMod is None:
            continue
        if codeMod.__name__ != module:
            continue
        if code.co_name in ('nextToken', '<module>'):
            continue
        rules.append(code.co_name)
    return rules