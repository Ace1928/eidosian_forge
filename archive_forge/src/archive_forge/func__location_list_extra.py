from collections import defaultdict
from .constants import *
from .dwarf_expr import DWARFExprParser
from .die import DIE
from ..common.utils import preserve_stream_pos, dwarf_assert, bytes2str
from .callframe import instruction_name, CIE, FDE
def _location_list_extra(attr, die, section_offset):
    if attr.form in ('DW_FORM_data4', 'DW_FORM_data8', 'DW_FORM_sec_offset'):
        return '(location list)'
    else:
        return describe_DWARF_expr(attr.value, die.cu.structs, die.cu.cu_offset)