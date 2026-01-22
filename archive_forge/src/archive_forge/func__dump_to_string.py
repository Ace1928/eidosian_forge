from collections import defaultdict
from .constants import *
from .dwarf_expr import DWARFExprParser
from .die import DIE
from ..common.utils import preserve_stream_pos, dwarf_assert, bytes2str
from .callframe import instruction_name, CIE, FDE
def _dump_to_string(self, opcode, opcode_name, args, cu_offset=None):
    if cu_offset is None:
        cu_offset = 0
    if len(args) == 0:
        if opcode_name.startswith('DW_OP_reg'):
            regnum = int(opcode_name[9:])
            return '%s (%s)' % (opcode_name, describe_reg_name(regnum, _MACHINE_ARCH))
        else:
            return opcode_name
    elif opcode_name in self._ops_with_decimal_arg:
        if opcode_name.startswith('DW_OP_breg'):
            regnum = int(opcode_name[10:])
            return '%s (%s): %s' % (opcode_name, describe_reg_name(regnum, _MACHINE_ARCH), args[0])
        elif opcode_name.endswith('regx'):
            return '%s: %s (%s)' % (opcode_name, args[0], describe_reg_name(args[0], _MACHINE_ARCH))
        else:
            return '%s: %s' % (opcode_name, args[0])
    elif opcode_name in self._ops_with_hex_arg:
        return '%s: %x' % (opcode_name, args[0])
    elif opcode_name in self._ops_with_two_decimal_args:
        return '%s: %s %s' % (opcode_name, args[0], args[1])
    elif opcode_name in ('DW_OP_GNU_entry_value', 'DW_OP_entry_value'):
        return '%s: (%s)' % (opcode_name, ','.join([self._dump_to_string(deo.op, deo.op_name, deo.args, cu_offset) for deo in args[0]]))
    elif opcode_name == 'DW_OP_implicit_value':
        return '%s %s byte block: %s' % (opcode_name, len(args[0]), ''.join(['%x ' % b for b in args[0]]))
    elif opcode_name == 'DW_OP_GNU_parameter_ref':
        return '%s: <0x%x>' % (opcode_name, args[0] + cu_offset)
    elif opcode_name in ('DW_OP_GNU_implicit_pointer', 'DW_OP_implicit_pointer'):
        return '%s: <0x%x> %d' % (opcode_name, args[0], args[1])
    elif opcode_name in ('DW_OP_GNU_convert', 'DW_OP_convert'):
        return '%s <0x%x>' % (opcode_name, args[0] + cu_offset)
    elif opcode_name in ('DW_OP_GNU_deref_type', 'DW_OP_deref_type'):
        return '%s: %d <0x%x>' % (opcode_name, args[0], args[1] + cu_offset)
    elif opcode_name in ('DW_OP_GNU_const_type', 'DW_OP_const_type'):
        return '%s: <0x%x>  %d byte block: %s ' % (opcode_name, args[0] + cu_offset, len(args[1]), ' '.join(('%x' % b for b in args[1])))
    elif opcode_name in ('DW_OP_GNU_regval_type', 'DW_OP_regval_type'):
        return '%s: %d (%s) <0x%x>' % (opcode_name, args[0], describe_reg_name(args[0], _MACHINE_ARCH), args[1] + cu_offset)
    else:
        return '<unknown %s>' % opcode_name