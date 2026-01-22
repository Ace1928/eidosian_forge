from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
from antlr3.constants import DEFAULT_CHANNEL, EOF
from antlr3.tokens import Token, EOF_TOKEN
import six
from six import StringIO
class DeleteOp(ReplaceOp):
    """
    @brief Internal helper class.
    """

    def __init__(self, stream, first, last):
        ReplaceOp.__init__(self, stream, first, last, None)

    def toString(self):
        return '<DeleteOp@%d..%d>' % (self.index, self.lastIndex)
    __str__ = toString
    __repr__ = toString