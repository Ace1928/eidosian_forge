import copy
from collections import namedtuple
from ..common.utils import (
from ..construct import Struct, Switch
from .enums import DW_EH_encoding_flags
from .structs import DWARFStructs
from .constants import *
class CFIEntry(object):
    """ A common base class for CFI entries.
        Contains a header and a list of instructions (CallFrameInstruction).
        offset: the offset of this entry from the beginning of the section
        cie: for FDEs, a CIE pointer is required
        augmentation_dict: Augmentation data as a parsed struct (dict): see
            CallFrameInfo._parse_cie_augmentation and
            http://www.airs.com/blog/archives/460.
        augmentation_bytes: Augmentation data as a chain of bytes: see
            CallFrameInfo._parse_cie_augmentation and
            http://www.airs.com/blog/archives/460.
    """

    def __init__(self, header, structs, instructions, offset, augmentation_dict=None, augmentation_bytes=b'', cie=None):
        self.header = header
        self.structs = structs
        self.instructions = instructions
        self.offset = offset
        self.cie = cie
        self._decoded_table = None
        self.augmentation_dict = augmentation_dict if augmentation_dict else {}
        self.augmentation_bytes = augmentation_bytes

    def get_decoded(self):
        """ Decode the CFI contained in this entry and return a
            DecodedCallFrameTable object representing it. See the documentation
            of that class to understand how to interpret the decoded table.
        """
        if self._decoded_table is None:
            self._decoded_table = self._decode_CFI_table()
        return self._decoded_table

    def __getitem__(self, name):
        """ Implement dict-like access to header entries
        """
        return self.header[name]

    def _decode_CFI_table(self):
        """ Decode the instructions contained in the given CFI entry and return
            a DecodedCallFrameTable.
        """
        if isinstance(self, CIE):
            cie = self
            cur_line = dict(pc=0, cfa=CFARule(reg=None, offset=0))
            reg_order = []
        else:
            cie = self.cie
            cie_decoded_table = cie.get_decoded()
            if len(cie_decoded_table.table) > 0:
                last_line_in_CIE = copy.copy(cie_decoded_table.table[-1])
                cur_line = copy.copy(last_line_in_CIE)
            else:
                cur_line = dict(cfa=CFARule(reg=None, offset=0))
            cur_line['pc'] = self['initial_location']
            reg_order = copy.copy(cie_decoded_table.reg_order)
        table = []
        line_stack = []

        def _add_to_order(regnum):
            if regnum not in reg_order:
                reg_order.append(regnum)
        for instr in self.instructions:
            name = instruction_name(instr.opcode)
            if name == 'DW_CFA_set_loc':
                table.append(copy.copy(cur_line))
                cur_line['pc'] = instr.args[0]
            elif name in ('DW_CFA_advance_loc1', 'DW_CFA_advance_loc2', 'DW_CFA_advance_loc4', 'DW_CFA_advance_loc'):
                table.append(copy.copy(cur_line))
                cur_line['pc'] += instr.args[0] * cie['code_alignment_factor']
            elif name == 'DW_CFA_def_cfa':
                cur_line['cfa'] = CFARule(reg=instr.args[0], offset=instr.args[1])
            elif name == 'DW_CFA_def_cfa_sf':
                cur_line['cfa'] = CFARule(reg=instr.args[0], offset=instr.args[1] * cie['code_alignment_factor'])
            elif name == 'DW_CFA_def_cfa_register':
                cur_line['cfa'] = CFARule(reg=instr.args[0], offset=cur_line['cfa'].offset)
            elif name == 'DW_CFA_def_cfa_offset':
                cur_line['cfa'] = CFARule(reg=cur_line['cfa'].reg, offset=instr.args[0])
            elif name == 'DW_CFA_def_cfa_expression':
                cur_line['cfa'] = CFARule(expr=instr.args[0])
            elif name == 'DW_CFA_undefined':
                _add_to_order(instr.args[0])
                cur_line[instr.args[0]] = RegisterRule(RegisterRule.UNDEFINED)
            elif name == 'DW_CFA_same_value':
                _add_to_order(instr.args[0])
                cur_line[instr.args[0]] = RegisterRule(RegisterRule.SAME_VALUE)
            elif name in ('DW_CFA_offset', 'DW_CFA_offset_extended', 'DW_CFA_offset_extended_sf'):
                _add_to_order(instr.args[0])
                cur_line[instr.args[0]] = RegisterRule(RegisterRule.OFFSET, instr.args[1] * cie['data_alignment_factor'])
            elif name in ('DW_CFA_val_offset', 'DW_CFA_val_offset_sf'):
                _add_to_order(instr.args[0])
                cur_line[instr.args[0]] = RegisterRule(RegisterRule.VAL_OFFSET, instr.args[1] * cie['data_alignment_factor'])
            elif name == 'DW_CFA_register':
                _add_to_order(instr.args[0])
                cur_line[instr.args[0]] = RegisterRule(RegisterRule.REGISTER, instr.args[1])
            elif name == 'DW_CFA_expression':
                _add_to_order(instr.args[0])
                cur_line[instr.args[0]] = RegisterRule(RegisterRule.EXPRESSION, instr.args[1])
            elif name == 'DW_CFA_val_expression':
                _add_to_order(instr.args[0])
                cur_line[instr.args[0]] = RegisterRule(RegisterRule.VAL_EXPRESSION, instr.args[1])
            elif name in ('DW_CFA_restore', 'DW_CFA_restore_extended'):
                _add_to_order(instr.args[0])
                dwarf_assert(isinstance(self, FDE), '%s instruction must be in a FDE' % name)
                if instr.args[0] in last_line_in_CIE:
                    cur_line[instr.args[0]] = last_line_in_CIE[instr.args[0]]
                else:
                    cur_line.pop(instr.args[0], None)
            elif name == 'DW_CFA_remember_state':
                line_stack.append(copy.deepcopy(cur_line))
            elif name == 'DW_CFA_restore_state':
                pc = cur_line['pc']
                cur_line = line_stack.pop()
                cur_line['pc'] = pc
        if cur_line['cfa'].reg is not None or len(cur_line) > 2:
            table.append(cur_line)
        return DecodedCallFrameTable(table=table, reg_order=reg_order)