import re
import sys
from weakref import ReferenceType
import weakref
from debian._util import resolve_ref, _strI
from debian._deb822_repro._util import BufferingIterator
class Deb822CommaToken(Deb822SeparatorToken):
    """Used by the comma-separated list value parsers to denote a comma between two value tokens."""
    __slots__ = ()

    def __init__(self):
        super().__init__(',')