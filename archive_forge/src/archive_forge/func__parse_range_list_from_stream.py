import os
from collections import namedtuple
from ..common.utils import struct_parse
from ..common.exceptions import DWARFError
from .dwarf_util import _iter_CUs_in_section
def _parse_range_list_from_stream(self, cu):
    if self.version >= 5:
        return list((entry_translate[entry.entry_type](entry, cu) for entry in struct_parse(self.structs.Dwarf_rnglists_entries, self.stream)))
    else:
        lst = []
        while True:
            entry_offset = self.stream.tell()
            begin_offset = struct_parse(self.structs.Dwarf_target_addr(''), self.stream)
            end_offset = struct_parse(self.structs.Dwarf_target_addr(''), self.stream)
            if begin_offset == 0 and end_offset == 0:
                break
            elif begin_offset == self._max_addr:
                lst.append(BaseAddressEntry(entry_offset=entry_offset, base_address=end_offset))
            else:
                lst.append(RangeEntry(entry_offset=entry_offset, entry_length=self.stream.tell() - entry_offset, begin_offset=begin_offset, end_offset=end_offset, is_absolute=False))
        return lst