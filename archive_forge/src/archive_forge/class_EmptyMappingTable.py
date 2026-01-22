import stringprep
from encodings import idna
from itertools import chain
from unicodedata import ucd_3_2_0 as unicodedata
from zope.interface import Interface, implementer
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
@implementer(IMappingTable)
class EmptyMappingTable:

    def __init__(self, in_table_function):
        self._in_table_function = in_table_function

    def map(self, c):
        if self._in_table_function(c):
            return None
        else:
            return c