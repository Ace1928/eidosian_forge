import stringprep
from encodings import idna
from itertools import chain
from unicodedata import ucd_3_2_0 as unicodedata
from zope.interface import Interface, implementer
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
@implementer(ILookupTable)
class LookupTable:

    def __init__(self, table):
        self._table = table

    def lookup(self, c):
        return c in self._table