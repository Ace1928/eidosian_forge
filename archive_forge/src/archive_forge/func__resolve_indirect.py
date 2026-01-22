from collections import namedtuple, OrderedDict
import os
from ..common.exceptions import DWARFError
from ..common.utils import bytes2str, struct_parse, preserve_stream_pos
from .enums import DW_FORM_raw2name
from .dwarf_util import _resolve_via_offset_table, _get_base_offset
def _resolve_indirect(self):
    structs = self.cu.structs
    length = 1
    real_form_code = struct_parse(structs.Dwarf_uleb128(''), self.stream)
    while True:
        try:
            real_form = DW_FORM_raw2name[real_form_code]
        except KeyError as err:
            raise DWARFError('Found DW_FORM_indirect with unknown real form 0x%x' % real_form_code)
        raw_value = struct_parse(structs.Dwarf_dw_form[real_form], self.stream)
        if real_form != 'DW_FORM_indirect':
            return (real_form, raw_value, length)
        else:
            length += 1
            real_form_code = raw_value