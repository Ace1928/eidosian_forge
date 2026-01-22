from collections import namedtuple
from ..common.exceptions import ELFRelocationError
from ..common.utils import elf_assert, struct_parse
from .sections import Section
from .enums import (
from ..construct import Container
class RelrRelocationTable(object):
    """ RELR compressed relocation table. This stores relative relocations
        in a compressed format. An entry with an even value serves as an
        'anchor' that defines a base address. Following this entry are one or
        more bitmaps for consecutive addresses after the anchor which determine
        if the corresponding relocation exists (if the bit is 1) or if it is
        skipped. Addends are stored at the respective addresses (as in REL
        relocations).
    """

    def __init__(self, elffile, offset, size, entrysize):
        self._elffile = elffile
        self._offset = offset
        self._size = size
        self._relr_struct = self._elffile.structs.Elf_Relr
        self._entrysize = self._relr_struct.sizeof()
        self._cached_relocations = None
        elf_assert(self._entrysize == entrysize, 'Expected RELR entry size to be %s, got %s' % (self._entrysize, entrysize))

    def iter_relocations(self):
        """ Yield all the relocations in the section
        """
        if self._size == 0:
            return []
        limit = self._offset + self._size
        relr = self._offset
        base = None
        while relr < limit:
            entry = struct_parse(self._relr_struct, self._elffile.stream, stream_pos=relr)
            entry_offset = entry['r_offset']
            if entry_offset & 1 == 0:
                base = entry_offset
                base += self._entrysize
                yield Relocation(entry, self._elffile)
            else:
                elf_assert(base is not None, 'RELR bitmap without base address')
                i = 0
                while True:
                    entry_offset = entry_offset >> 1
                    if entry_offset == 0:
                        break
                    if entry_offset & 1 != 0:
                        calc_offset = base + i * self._entrysize
                        yield Relocation(Container(r_offset=calc_offset), self._elffile)
                    i += 1
                base += (8 * self._entrysize - 1) * self._elffile.structs.Elf_addr('').sizeof()
            relr += self._entrysize

    def num_relocations(self):
        """ Number of relocations in the section
        """
        if self._cached_relocations is None:
            self._cached_relocations = list(self.iter_relocations())
        return len(self._cached_relocations)

    def get_relocation(self, n):
        """ Get the relocation at index #n from the section (Relocation object)
        """
        if self._cached_relocations is None:
            self._cached_relocations = list(self.iter_relocations())
        return self._cached_relocations[n]