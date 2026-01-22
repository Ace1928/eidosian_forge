from logging.config import valid_ident
from ..construct import (
from ..common.construct_utils import (RepeatUntilExcluding, ULEB128, SLEB128,
from .enums import *
def _create_cu_header(self):
    dwarfv4_CU_header = Struct('', self.Dwarf_offset('debug_abbrev_offset'), self.Dwarf_uint8('address_size'))
    dwarfv5_CP_CU_header = Struct('', self.Dwarf_uint8('address_size'), self.Dwarf_offset('debug_abbrev_offset'))
    dwarfv5_SS_CU_header = Struct('', self.Dwarf_uint8('address_size'), self.Dwarf_offset('debug_abbrev_offset'), self.Dwarf_uint64('dwo_id'))
    dwarfv5_TS_CU_header = Struct('', self.Dwarf_uint8('address_size'), self.Dwarf_offset('debug_abbrev_offset'), self.Dwarf_uint64('type_signature'), self.Dwarf_offset('type_offset'))
    dwarfv5_CU_header = Struct('', Enum(self.Dwarf_uint8('unit_type'), **ENUM_DW_UT), Embed(Switch('', lambda ctx: ctx.unit_type, {'DW_UT_compile': dwarfv5_CP_CU_header, 'DW_UT_partial': dwarfv5_CP_CU_header, 'DW_UT_skeleton': dwarfv5_SS_CU_header, 'DW_UT_split_compile': dwarfv5_SS_CU_header, 'DW_UT_type': dwarfv5_TS_CU_header, 'DW_UT_split_type': dwarfv5_TS_CU_header})))
    self.Dwarf_CU_header = Struct('Dwarf_CU_header', self.Dwarf_initial_length('unit_length'), self.Dwarf_uint16('version'), IfThenElse('', lambda ctx: ctx['version'] >= 5, Embed(dwarfv5_CU_header), Embed(dwarfv4_CU_header)))