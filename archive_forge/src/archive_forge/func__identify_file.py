import io
from io import BytesIO
import os
import struct
import zlib
from ..common.exceptions import ELFError, ELFParseError
from ..common.utils import struct_parse, elf_assert
from .structs import ELFStructs
from .sections import (
from .dynamic import DynamicSection, DynamicSegment
from .relocation import (RelocationSection, RelocationHandler,
from .gnuversions import (
from .segments import Segment, InterpSegment, NoteSegment
from ..dwarf.dwarfinfo import DWARFInfo, DebugSectionDescriptor, DwarfConfig
from ..ehabi.ehabiinfo import EHABIInfo
from .hash import ELFHashSection, GNUHashSection
from .constants import SHN_INDICES
def _identify_file(self):
    """ Verify the ELF file and identify its class and endianness.
        """
    self.stream.seek(0)
    magic = self.stream.read(4)
    elf_assert(magic == b'\x7fELF', 'Magic number does not match')
    ei_class = self.stream.read(1)
    if ei_class == b'\x01':
        self.elfclass = 32
    elif ei_class == b'\x02':
        self.elfclass = 64
    else:
        raise ELFError('Invalid EI_CLASS %s' % repr(ei_class))
    ei_data = self.stream.read(1)
    if ei_data == b'\x01':
        self.little_endian = True
    elif ei_data == b'\x02':
        self.little_endian = False
    else:
        raise ELFError('Invalid EI_DATA %s' % repr(ei_data))