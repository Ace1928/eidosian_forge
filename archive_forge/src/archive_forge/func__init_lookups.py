from collections import defaultdict
from .constants import *
from .dwarf_expr import DWARFExprParser
from .die import DIE
from ..common.utils import preserve_stream_pos, dwarf_assert, bytes2str
from .callframe import instruction_name, CIE, FDE
def _init_lookups(self):
    self._ops_with_decimal_arg = set(['DW_OP_const1u', 'DW_OP_const1s', 'DW_OP_const2u', 'DW_OP_const2s', 'DW_OP_const4u', 'DW_OP_const4s', 'DW_OP_const8u', 'DW_OP_const8s', 'DW_OP_constu', 'DW_OP_consts', 'DW_OP_pick', 'DW_OP_plus_uconst', 'DW_OP_bra', 'DW_OP_skip', 'DW_OP_fbreg', 'DW_OP_piece', 'DW_OP_deref_size', 'DW_OP_xderef_size', 'DW_OP_regx'])
    for n in range(0, 32):
        self._ops_with_decimal_arg.add('DW_OP_breg%s' % n)
    self._ops_with_two_decimal_args = set(['DW_OP_bregx', 'DW_OP_bit_piece'])
    self._ops_with_hex_arg = set(['DW_OP_addr', 'DW_OP_call2', 'DW_OP_call4', 'DW_OP_call_ref'])