from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.constants import UP, DOWN, EOF, INVALID_TOKEN_TYPE
from antlr3.exceptions import MismatchedTreeNodeException, \
from antlr3.recognizers import BaseRecognizer, RuleReturnScope
from antlr3.streams import IntStream
from antlr3.tokens import CommonToken, Token, INVALID_TOKEN
import six
from six.moves import range
def _fillBuffer(self, t):
    nil = self.adaptor.isNil(t)
    if not nil:
        self.nodes.append(t)
    n = self.adaptor.getChildCount(t)
    if not nil and n > 0:
        self.addNavigationNode(DOWN)
    for c in range(n):
        self._fillBuffer(self.adaptor.getChild(t, c))
    if not nil and n > 0:
        self.addNavigationNode(UP)