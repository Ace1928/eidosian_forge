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
def toStrings(self, tokens):
    """A convenience method for use most often with template rewrites.

        Convert a List<Token> to List<String>
        """
    if tokens is None:
        return None
    return [token.text for token in tokens]