import os
from collections import namedtuple
from bisect import bisect_right
from ..construct.lib.container import Container
from ..common.exceptions import DWARFError
from ..common.utils import (struct_parse, dwarf_assert,
from .structs import DWARFStructs
from .compileunit import CompileUnit
from .abbrevtable import AbbrevTable
from .lineprogram import LineProgram
from .callframe import CallFrameInfo
from .locationlists import LocationLists, LocationListsPair
from .ranges import RangeLists, RangeListsPair
from .aranges import ARanges
from .namelut import NameLUT
from .dwarf_util import _get_base_offset
def _parse_CU_at_offset(self, offset):
    """ Parse and return a CU at the given offset in the debug_info stream.
        """
    initial_length = struct_parse(self.structs.Dwarf_uint32(''), self.debug_info_sec.stream, offset)
    dwarf_format = 64 if initial_length == 4294967295 else 32
    cu_structs = DWARFStructs(little_endian=self.config.little_endian, dwarf_format=dwarf_format, address_size=4, dwarf_version=2)
    cu_header = struct_parse(cu_structs.Dwarf_CU_header, self.debug_info_sec.stream, offset)
    cu_structs = DWARFStructs(little_endian=self.config.little_endian, dwarf_format=dwarf_format, address_size=cu_header['address_size'], dwarf_version=cu_header['version'])
    cu_die_offset = self.debug_info_sec.stream.tell()
    dwarf_assert(self._is_supported_version(cu_header['version']), "Expected supported DWARF version. Got '%s'" % cu_header['version'])
    return CompileUnit(header=cu_header, dwarfinfo=self, structs=cu_structs, cu_offset=offset, cu_die_offset=cu_die_offset)