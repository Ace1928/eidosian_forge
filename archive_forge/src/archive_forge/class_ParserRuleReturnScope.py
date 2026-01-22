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
class ParserRuleReturnScope(RuleReturnScope):
    """
    Rules that return more than a single value must return an object
    containing all the values.  Besides the properties defined in
    RuleLabelScope.predefinedRulePropertiesScope there may be user-defined
    return values.  This class simply defines the minimum properties that
    are always defined and methods to access the others that might be
    available depending on output option such as template and tree.

    Note text is not an actual property of the return value, it is computed
    from start and stop using the input stream's toString() method.  I
    could add a ctor to this so that we can pass in and store the input
    stream, but I'm not sure we want to do that.  It would seem to be undefined
    to get the .text property anyway if the rule matches tokens from multiple
    input streams.

    I do not use getters for fields of objects that are used simply to
    group values such as this aggregate.  The getters/setters are there to
    satisfy the superclass interface.
    """

    def __init__(self):
        self.start = None
        self.stop = None

    def getStart(self):
        return self.start

    def getStop(self):
        return self.stop