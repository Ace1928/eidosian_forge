from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
def _make_attributes(self):
    """ Create all attributes for this subsubsection except the first one
            which is the header.
        """
    end = self.offset + self.header.value
    self.stream.seek(self.attr_start)
    while self.stream.tell() != end:
        yield self.attribute(self.structs, self.stream)