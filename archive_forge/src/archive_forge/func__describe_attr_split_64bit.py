from collections import defaultdict
from .constants import *
from .dwarf_expr import DWARFExprParser
from .die import DIE
from ..common.utils import preserve_stream_pos, dwarf_assert, bytes2str
from .callframe import instruction_name, CIE, FDE
def _describe_attr_split_64bit(attr, die, section_offset):
    low_word = attr.value & 4294967295
    high_word = attr.value >> 32 & 4294967295
    return '%s %s' % (_format_hex(low_word), _format_hex(high_word))