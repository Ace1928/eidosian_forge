from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.constants import INVALID_TOKEN_TYPE
from antlr3.tokens import CommonToken
from antlr3.tree import CommonTree, CommonTreeAdaptor
import six
from six.moves import range
class TreePatternTreeAdaptor(CommonTreeAdaptor):
    """This adaptor creates TreePattern objects for use during scan()"""

    def createWithPayload(self, payload):
        return TreePattern(payload)