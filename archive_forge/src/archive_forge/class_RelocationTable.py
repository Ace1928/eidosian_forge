from collections import namedtuple
from ..common.exceptions import ELFRelocationError
from ..common.utils import elf_assert, struct_parse
from .sections import Section
from .enums import (
from ..construct import Container
class RelocationTable(object):
    """ Shared functionality between relocation sections and relocation tables
    """

    def __init__(self, elffile, offset, size, is_rela):
        self._stream = elffile.stream
        self._elffile = elffile
        self._elfstructs = elffile.structs
        self._size = size
        self._offset = offset
        self._is_rela = is_rela
        if is_rela:
            self.entry_struct = self._elfstructs.Elf_Rela
        else:
            self.entry_struct = self._elfstructs.Elf_Rel
        self.entry_size = self.entry_struct.sizeof()

    def is_RELA(self):
        """ Is this a RELA relocation section? If not, it's REL.
        """
        return self._is_rela

    def num_relocations(self):
        """ Number of relocations in the section
        """
        return self._size // self.entry_size

    def get_relocation(self, n):
        """ Get the relocation at index #n from the section (Relocation object)
        """
        entry_offset = self._offset + n * self.entry_size
        entry = struct_parse(self.entry_struct, self._stream, stream_pos=entry_offset)
        return Relocation(entry, self._elffile)

    def iter_relocations(self):
        """ Yield all the relocations in the section
        """
        for i in range(self.num_relocations()):
            yield self.get_relocation(i)