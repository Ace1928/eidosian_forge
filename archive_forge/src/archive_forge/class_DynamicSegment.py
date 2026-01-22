import itertools
from collections import defaultdict
from .hash import ELFHashTable, GNUHashTable
from .sections import Section, Symbol
from .enums import ENUM_D_TAG
from .segments import Segment
from .relocation import RelocationTable, RelrRelocationTable
from ..common.exceptions import ELFError
from ..common.utils import elf_assert, struct_parse, parse_cstring_from_stream
class DynamicSegment(Segment, Dynamic):
    """ ELF dynamic table segment.  Knows how to process the list of tags.
    """

    def __init__(self, header, stream, elffile):
        stringtable = None
        for section in elffile.iter_sections():
            if isinstance(section, DynamicSection) and section['sh_offset'] == header['p_offset']:
                stringtable = elffile.get_section(section['sh_link'])
                break
        Segment.__init__(self, header, stream)
        Dynamic.__init__(self, stream, elffile, stringtable, self['p_offset'], self['p_filesz'] == 0)
        self._symbol_size = self.elfstructs.Elf_Sym.sizeof()
        self._num_symbols = None
        self._symbol_name_map = None

    def num_symbols(self):
        """ Number of symbols in the table recovered from DT_SYMTAB
        """
        if self._num_symbols is not None:
            return self._num_symbols
        _, gnu_hash_offset = self.get_table_offset('DT_GNU_HASH')
        if gnu_hash_offset is not None:
            hash_section = GNUHashTable(self.elffile, gnu_hash_offset, self)
            self._num_symbols = hash_section.get_number_of_symbols()
        if self._num_symbols is None:
            _, hash_offset = self.get_table_offset('DT_HASH')
            if hash_offset is not None:
                hash_section = ELFHashTable(self.elffile, hash_offset, self)
                self._num_symbols = hash_section.get_number_of_symbols()
        if self._num_symbols is None:
            tab_ptr, tab_offset = self.get_table_offset('DT_SYMTAB')
            if tab_ptr is None or tab_offset is None:
                raise ELFError('Segment does not contain DT_SYMTAB.')
            nearest_ptr = None
            for tag in self.iter_tags():
                tag_ptr = tag['d_ptr']
                if tag['d_tag'] == 'DT_SYMENT':
                    if self._symbol_size != tag['d_val']:
                        raise ELFError('DT_SYMENT (%d) != Elf_Sym (%d).' % (tag['d_val'], self._symbol_size))
                if tag_ptr > tab_ptr and (nearest_ptr is None or nearest_ptr > tag_ptr):
                    nearest_ptr = tag_ptr
            if nearest_ptr is None:
                for segment in self.elffile.iter_segments():
                    if segment['p_vaddr'] <= tab_ptr and tab_ptr <= segment['p_vaddr'] + segment['p_filesz']:
                        nearest_ptr = segment['p_vaddr'] + segment['p_filesz']
            end_ptr = nearest_ptr
            self._num_symbols = (end_ptr - tab_ptr) // self._symbol_size
        if self._num_symbols is None:
            raise ELFError('Cannot determine the end of DT_SYMTAB.')
        return self._num_symbols

    def get_symbol(self, index):
        """ Get the symbol at index #index from the table (Symbol object)
        """
        tab_ptr, tab_offset = self.get_table_offset('DT_SYMTAB')
        if tab_ptr is None or tab_offset is None:
            raise ELFError('Segment does not contain DT_SYMTAB.')
        symbol = struct_parse(self.elfstructs.Elf_Sym, self._stream, stream_pos=tab_offset + index * self._symbol_size)
        string_table = self._get_stringtable()
        symbol_name = string_table.get_string(symbol['st_name'])
        return Symbol(symbol, symbol_name)

    def get_symbol_by_name(self, name):
        """ Get a symbol(s) by name. Return None if no symbol by the given name
            exists.
        """
        if self._symbol_name_map is None:
            self._symbol_name_map = defaultdict(list)
            for i, sym in enumerate(self.iter_symbols()):
                self._symbol_name_map[sym.name].append(i)
        symnums = self._symbol_name_map.get(name)
        return [self.get_symbol(i) for i in symnums] if symnums else None

    def iter_symbols(self):
        """ Yield all symbols in this dynamic segment. The symbols are usually
            the same as returned by SymbolTableSection.iter_symbols. However,
            in stripped binaries, SymbolTableSection might have been removed.
            This method reads from the mandatory dynamic tag DT_SYMTAB.
        """
        for i in range(self.num_symbols()):
            yield self.get_symbol(i)