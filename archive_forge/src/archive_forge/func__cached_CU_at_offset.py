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
def _cached_CU_at_offset(self, offset):
    """ Return the CU with unit header at the given offset into the
            debug_info section from the cache.  If not present, the unit is
            header is parsed and the object is installed in the cache.

            offset:
                The offset of the unit header in the .debug_info section
                to of the unit to fetch from the cache.

            See get_CU_at().
        """
    i = bisect_right(self._cu_offsets_map, offset)
    if i >= 1 and offset == self._cu_offsets_map[i - 1]:
        return self._cu_cache[i - 1]
    cu = self._parse_CU_at_offset(offset)
    self._cu_offsets_map.insert(i, offset)
    self._cu_cache.insert(i, cu)
    return cu