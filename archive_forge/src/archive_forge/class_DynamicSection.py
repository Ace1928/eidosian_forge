import itertools
from collections import defaultdict
from .hash import ELFHashTable, GNUHashTable
from .sections import Section, Symbol
from .enums import ENUM_D_TAG
from .segments import Segment
from .relocation import RelocationTable, RelrRelocationTable
from ..common.exceptions import ELFError
from ..common.utils import elf_assert, struct_parse, parse_cstring_from_stream
class DynamicSection(Section, Dynamic):
    """ ELF dynamic table section.  Knows how to process the list of tags.
    """

    def __init__(self, header, name, elffile):
        Section.__init__(self, header, name, elffile)
        stringtable = elffile.get_section(header['sh_link'])
        Dynamic.__init__(self, self.stream, self.elffile, stringtable, self['sh_offset'], self['sh_type'] == 'SHT_NOBITS')