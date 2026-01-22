from logging.config import valid_ident
from ..construct import (
from ..common.construct_utils import (RepeatUntilExcluding, ULEB128, SLEB128,
from .enums import *
def _create_callframe_entry_headers(self):
    self.Dwarf_CIE_header = Struct('Dwarf_CIE_header', self.Dwarf_initial_length('length'), self.Dwarf_offset('CIE_id'), self.Dwarf_uint8('version'), CString('augmentation'), self.Dwarf_uleb128('code_alignment_factor'), self.Dwarf_sleb128('data_alignment_factor'), self.Dwarf_uleb128('return_address_register'))
    self.EH_CIE_header = self.Dwarf_CIE_header
    if self.dwarf_version == 4:
        self.Dwarf_CIE_header = Struct('Dwarf_CIE_header', self.Dwarf_initial_length('length'), self.Dwarf_offset('CIE_id'), self.Dwarf_uint8('version'), CString('augmentation'), self.Dwarf_uint8('address_size'), self.Dwarf_uint8('segment_size'), self.Dwarf_uleb128('code_alignment_factor'), self.Dwarf_sleb128('data_alignment_factor'), self.Dwarf_uleb128('return_address_register'))
    self.Dwarf_FDE_header = Struct('Dwarf_FDE_header', self.Dwarf_initial_length('length'), self.Dwarf_offset('CIE_pointer'), self.Dwarf_target_addr('initial_location'), self.Dwarf_target_addr('address_range'))