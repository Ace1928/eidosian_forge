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
def _make_gnu_verdef_section(self, section_header, name):
    """ Create a GNUVerDefSection
        """
    linked_strtab_index = section_header['sh_link']
    strtab_section = self.get_section(linked_strtab_index)
    return GNUVerDefSection(section_header, name, elffile=self, stringtable=strtab_section)