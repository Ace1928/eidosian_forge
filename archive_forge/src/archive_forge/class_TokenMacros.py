import sys
import re
import operator
import typing
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
class TokenMacros:
    unicode_escape = '\\\\([0-9a-f]{1,6})(?:\\r\\n|[ \\n\\r\\t\\f])?'
    escape = unicode_escape + '|\\\\[^\\n\\r\\f0-9a-f]'
    string_escape = '\\\\(?:\\n|\\r\\n|\\r|\\f)|' + escape
    nonascii = '[^\\0-\\177]'
    nmchar = '[_a-z0-9-]|%s|%s' % (escape, nonascii)
    nmstart = '[_a-z]|%s|%s' % (escape, nonascii)