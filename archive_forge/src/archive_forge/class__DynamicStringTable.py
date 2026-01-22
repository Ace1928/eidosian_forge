import itertools
from collections import defaultdict
from .hash import ELFHashTable, GNUHashTable
from .sections import Section, Symbol
from .enums import ENUM_D_TAG
from .segments import Segment
from .relocation import RelocationTable, RelrRelocationTable
from ..common.exceptions import ELFError
from ..common.utils import elf_assert, struct_parse, parse_cstring_from_stream
class _DynamicStringTable(object):
    """ Bare string table based on values found via ELF dynamic tags and
        loadable segments only.  Good enough for get_string() only.
    """

    def __init__(self, stream, table_offset):
        self._stream = stream
        self._table_offset = table_offset

    def get_string(self, offset):
        """ Get the string stored at the given offset in this string table.
        """
        s = parse_cstring_from_stream(self._stream, self._table_offset + offset)
        return s.decode('utf-8') if s else ''