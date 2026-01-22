from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
class SymbolTableIndexSection(Section):
    """ A section containing the section header table indices corresponding
        to symbols in the linked symbol table. This section has to exist if the
        symbol table contains an entry with a section header index set to
        SHN_XINDEX (0xffff). The format of the section is described at
        https://refspecs.linuxfoundation.org/elf/gabi4+/ch4.sheader.html
    """

    def __init__(self, header, name, elffile, symboltable):
        super(SymbolTableIndexSection, self).__init__(header, name, elffile)
        self.symboltable = symboltable

    def get_section_index(self, n):
        """ Get the section header table index for the symbol with index #n.
            The section contains an array of Elf32_word values with one entry
            for every symbol in the associated symbol table.
        """
        return struct_parse(self.elffile.structs.Elf_word(''), self.stream, self['sh_offset'] + n * self['sh_entsize'])