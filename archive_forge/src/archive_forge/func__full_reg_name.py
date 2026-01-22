from collections import defaultdict
from .constants import *
from .dwarf_expr import DWARFExprParser
from .die import DIE
from ..common.utils import preserve_stream_pos, dwarf_assert, bytes2str
from .callframe import instruction_name, CIE, FDE
def _full_reg_name(regnum):
    regname = describe_reg_name(regnum, _MACHINE_ARCH, False)
    if regname:
        return 'r%s (%s)' % (regnum, regname)
    else:
        return 'r%s' % regnum