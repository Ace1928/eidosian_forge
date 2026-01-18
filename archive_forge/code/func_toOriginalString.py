from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
from antlr3.constants import DEFAULT_CHANNEL, EOF
from antlr3.tokens import Token, EOF_TOKEN
import six
from six import StringIO
def toOriginalString(self, start=None, end=None):
    if start is None:
        start = self.MIN_TOKEN_INDEX
    if end is None:
        end = self.size() - 1
    buf = StringIO()
    i = start
    while i >= self.MIN_TOKEN_INDEX and i <= end and (i < len(self.tokens)):
        buf.write(self.get(i).text)
        i += 1
    return buf.getvalue()