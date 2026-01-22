import re
import six
from genshi.core import Attrs, QName, stripentities
from genshi.core import END, START, TEXT, COMMENT
def _repl(match):
    t = match.group(1)
    if t:
        return six.unichr(int(t, 16))
    t = match.group(2)
    if t == '\\':
        return '\\\\'
    else:
        return t