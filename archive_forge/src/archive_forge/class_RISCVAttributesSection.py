from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
class RISCVAttributesSection(AttributesSection):
    """ ELF .riscv.attributes section.
    """

    def __init__(self, header, name, elffile):
        super(RISCVAttributesSection, self).__init__(header, name, elffile, RISCVAttributesSubsection)