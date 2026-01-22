import itertools
from collections import defaultdict
from .hash import ELFHashTable, GNUHashTable
from .sections import Section, Symbol
from .enums import ENUM_D_TAG
from .segments import Segment
from .relocation import RelocationTable, RelrRelocationTable
from ..common.exceptions import ELFError
from ..common.utils import elf_assert, struct_parse, parse_cstring_from_stream
class Dynamic(object):
    """ Shared functionality between dynamic sections and segments.
    """

    def __init__(self, stream, elffile, stringtable, position, empty):
        """
        stream:
            The file-like object from which to load data

        elffile:
            The parent elffile object

        stringtable:
            A stringtable reference to use for parsing string references in
            entries

        position:
            The file offset of the dynamic segment/section

        empty:
            Whether this is a degenerate case with zero entries. Normally, every
            dynamic table will have at least one entry, the DT_NULL terminator.
        """
        self.elffile = elffile
        self.elfstructs = elffile.structs
        self._stream = stream
        self._num_tags = -1 if not empty else 0
        self._offset = position
        self._tagsize = self.elfstructs.Elf_Dyn.sizeof()
        self._empty = empty
        self._stringtable = stringtable

    def get_table_offset(self, tag_name):
        """ Return the virtual address and file offset of a dynamic table.
        """
        ptr = None
        for tag in self._iter_tags(type=tag_name):
            ptr = tag['d_ptr']
            break
        offset = None
        if ptr:
            offset = next(self.elffile.address_offsets(ptr), None)
        return (ptr, offset)

    def _get_stringtable(self):
        """ Return a string table for looking up dynamic tag related strings.

            This won't be a "full" string table object, but will at least
            support the get_string() function.
        """
        if self._stringtable:
            return self._stringtable
        _, table_offset = self.get_table_offset('DT_STRTAB')
        if table_offset is not None:
            self._stringtable = _DynamicStringTable(self._stream, table_offset)
            return self._stringtable
        self._stringtable = self.elffile.get_section_by_name('.dynstr')
        return self._stringtable

    def _iter_tags(self, type=None):
        """ Yield all raw tags (limit to |type| if specified)
        """
        if self._empty:
            return
        for n in itertools.count():
            tag = self._get_tag(n)
            if type is None or tag['d_tag'] == type:
                yield tag
            if tag['d_tag'] == 'DT_NULL':
                break

    def iter_tags(self, type=None):
        """ Yield all tags (limit to |type| if specified)
        """
        for tag in self._iter_tags(type=type):
            yield DynamicTag(tag, self._get_stringtable())

    def _get_tag(self, n):
        """ Get the raw tag at index #n from the file
        """
        if self._num_tags != -1 and n >= self._num_tags:
            raise IndexError(n)
        offset = self._offset + n * self._tagsize
        return struct_parse(self.elfstructs.Elf_Dyn, self._stream, stream_pos=offset)

    def get_tag(self, n):
        """ Get the tag at index #n from the file (DynamicTag object)
        """
        return DynamicTag(self._get_tag(n), self._get_stringtable())

    def num_tags(self):
        """ Number of dynamic tags in the file, including the DT_NULL tag
        """
        if self._num_tags != -1:
            return self._num_tags
        for n in itertools.count():
            tag = self.get_tag(n)
            if tag.entry.d_tag == 'DT_NULL':
                self._num_tags = n + 1
                return self._num_tags

    def get_relocation_tables(self):
        """ Load all available relocation tables from DYNAMIC tags.

            Returns a dictionary mapping found table types (REL, RELA,
            RELR, JMPREL) to RelocationTable objects.
        """
        result = {}
        if list(self.iter_tags('DT_REL')):
            result['REL'] = RelocationTable(self.elffile, self.get_table_offset('DT_REL')[1], next(self.iter_tags('DT_RELSZ'))['d_val'], False)
            relentsz = next(self.iter_tags('DT_RELENT'))['d_val']
            elf_assert(result['REL'].entry_size == relentsz, 'Expected DT_RELENT to be %s' % relentsz)
        if list(self.iter_tags('DT_RELA')):
            result['RELA'] = RelocationTable(self.elffile, self.get_table_offset('DT_RELA')[1], next(self.iter_tags('DT_RELASZ'))['d_val'], True)
            relentsz = next(self.iter_tags('DT_RELAENT'))['d_val']
            elf_assert(result['RELA'].entry_size == relentsz, 'Expected DT_RELAENT to be %s' % relentsz)
        if list(self.iter_tags('DT_RELR')):
            result['RELR'] = RelrRelocationTable(self.elffile, self.get_table_offset('DT_RELR')[1], next(self.iter_tags('DT_RELRSZ'))['d_val'], next(self.iter_tags('DT_RELRENT'))['d_val'])
        if list(self.iter_tags('DT_JMPREL')):
            result['JMPREL'] = RelocationTable(self.elffile, self.get_table_offset('DT_JMPREL')[1], next(self.iter_tags('DT_PLTRELSZ'))['d_val'], next(self.iter_tags('DT_PLTREL'))['d_val'] == ENUM_D_TAG['DT_RELA'])
        return result