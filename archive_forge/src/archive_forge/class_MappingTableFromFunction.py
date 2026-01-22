import stringprep
from encodings import idna
from itertools import chain
from unicodedata import ucd_3_2_0 as unicodedata
from zope.interface import Interface, implementer
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
@implementer(IMappingTable)
class MappingTableFromFunction:

    def __init__(self, map_table_function):
        self.map = map_table_function