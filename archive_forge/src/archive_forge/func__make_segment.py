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
def _make_segment(self, segment_header):
    """ Create a Segment object of the appropriate type
        """
    segtype = segment_header['p_type']
    if segtype == 'PT_INTERP':
        return InterpSegment(segment_header, self.stream)
    elif segtype == 'PT_DYNAMIC':
        return DynamicSegment(segment_header, self.stream, self)
    elif segtype == 'PT_NOTE':
        return NoteSegment(segment_header, self.stream, self)
    else:
        return Segment(segment_header, self.stream)