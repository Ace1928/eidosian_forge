import os
from collections import namedtuple
from ..common.exceptions import DWARFError
from ..common.utils import struct_parse
from .dwarf_util import _iter_CUs_in_section
def _parse_location_list_from_stream(self):
    lst = []
    while True:
        entry_offset = self.stream.tell()
        begin_offset = struct_parse(self.structs.Dwarf_target_addr(''), self.stream)
        end_offset = struct_parse(self.structs.Dwarf_target_addr(''), self.stream)
        if begin_offset == 0 and end_offset == 0:
            break
        elif begin_offset == self._max_addr:
            entry_length = self.stream.tell() - entry_offset
            lst.append(BaseAddressEntry(entry_offset=entry_offset, entry_length=entry_length, base_address=end_offset))
        else:
            expr_len = struct_parse(self.structs.Dwarf_uint16(''), self.stream)
            loc_expr = [struct_parse(self.structs.Dwarf_uint8(''), self.stream) for i in range(expr_len)]
            entry_length = self.stream.tell() - entry_offset
            lst.append(LocationEntry(entry_offset=entry_offset, entry_length=entry_length, begin_offset=begin_offset, end_offset=end_offset, loc_expr=loc_expr, is_absolute=False))
    return lst