from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
class ARMAttributesSubsubsection(AttributesSubsubsection):
    """ Subsubsection of an ELF .ARM.attributes section's subsection.
    """

    def __init__(self, stream, structs, offset):
        super(ARMAttributesSubsubsection, self).__init__(stream, structs, offset, ARMAttribute)