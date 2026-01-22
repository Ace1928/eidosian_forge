from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
from antlr3.constants import DEFAULT_CHANNEL, EOF
from antlr3.tokens import Token, EOF_TOKEN
import six
from six import StringIO
class RewriteOperation(object):
    """@brief Internal helper class."""

    def __init__(self, stream, index, text):
        self.stream = stream
        self.index = index
        self.text = text

    def execute(self, buf):
        """Execute the rewrite operation by possibly adding to the buffer.

        Return the index of the next token to operate on.
        """
        return self.index

    def toString(self):
        opName = self.__class__.__name__
        return '<%s@%d:"%s">' % (opName, self.index, self.text)
    __str__ = toString
    __repr__ = toString