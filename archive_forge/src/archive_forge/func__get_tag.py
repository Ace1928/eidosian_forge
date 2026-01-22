import itertools
from collections import defaultdict
from .hash import ELFHashTable, GNUHashTable
from .sections import Section, Symbol
from .enums import ENUM_D_TAG
from .segments import Segment
from .relocation import RelocationTable, RelrRelocationTable
from ..common.exceptions import ELFError
from ..common.utils import elf_assert, struct_parse, parse_cstring_from_stream
def _get_tag(self, n):
    """ Get the raw tag at index #n from the file
        """
    if self._num_tags != -1 and n >= self._num_tags:
        raise IndexError(n)
    offset = self._offset + n * self._tagsize
    return struct_parse(self.elfstructs.Elf_Dyn, self._stream, stream_pos=offset)