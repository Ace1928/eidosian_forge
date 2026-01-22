from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
class ARMAttributesSubsection(AttributesSubsection):
    """ Subsection of an ELF .ARM.attributes section.
    """

    def __init__(self, stream, structs, offset):
        super(ARMAttributesSubsection, self).__init__(stream, structs, offset, structs.Elf_Attr_Subsection_Header, ARMAttributesSubsubsection)