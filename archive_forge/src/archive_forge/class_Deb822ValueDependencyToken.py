import re
import sys
from weakref import ReferenceType
import weakref
from debian._util import resolve_ref, _strI
from debian._deb822_repro._util import BufferingIterator
class Deb822ValueDependencyToken(Deb822Token):
    """Package name, architecture name, a version number, or a profile name in a dependency field"""
    __slots__ = ()