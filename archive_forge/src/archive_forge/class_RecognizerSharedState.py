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
class RecognizerSharedState(object):
    """
    The set of fields needed by an abstract recognizer to recognize input
    and recover from errors etc...  As a separate state object, it can be
    shared among multiple grammars; e.g., when one grammar imports another.

    These fields are publically visible but the actual state pointer per
    parser is protected.
    """

    def __init__(self):
        self.following = []
        self.errorRecovery = False
        self.lastErrorIndex = -1
        self.backtracking = 0
        self.ruleMemo = None
        self.syntaxErrors = 0
        self.token = None
        self.tokenStartCharIndex = -1
        self.tokenStartLine = None
        self.tokenStartCharPositionInLine = None
        self.channel = None
        self.type = None
        self.text = None