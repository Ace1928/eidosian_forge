import os
from collections import namedtuple
from ..common.exceptions import DWARFError
from ..common.utils import struct_parse
from .dwarf_util import _iter_CUs_in_section
def _translate_entry_v5(self, entry, die):
    off = entry.entry_offset
    len = entry.entry_end_offset - off
    type = entry.entry_type
    if type == 'DW_LLE_base_address':
        return BaseAddressEntry(off, len, entry.address)
    elif type == 'DW_LLE_offset_pair':
        return LocationEntry(off, len, entry.start_offset, entry.end_offset, entry.loc_expr, False)
    elif type == 'DW_LLE_start_length':
        return LocationEntry(off, len, entry.start_address, entry.start_address + entry.length, entry.loc_expr, True)
    elif type == 'DW_LLE_start_end':
        return LocationEntry(off, len, entry.start_address, entry.end_address, entry.loc_expr, True)
    elif type == 'DW_LLE_default_location':
        return LocationEntry(off, len, -1, -1, entry.loc_expr, True)
    elif type in ('DW_LLE_base_addressx', 'DW_LLE_startx_endx', 'DW_LLE_startx_length'):
        raise NotImplementedError('Location list entry type %s is not supported yet' % (type,))
    else:
        raise DWARFError(False, 'Unknown DW_LLE code: %s' % (type,))