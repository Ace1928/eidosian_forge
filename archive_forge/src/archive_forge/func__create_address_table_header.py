from logging.config import valid_ident
from ..construct import (
from ..common.construct_utils import (RepeatUntilExcluding, ULEB128, SLEB128,
from .enums import *
def _create_address_table_header(self):
    self.Dwarf_address_table_header = Struct('Dwarf_address_table_header', self.Dwarf_initial_length('unit_length'), self.Dwarf_uint16('version'), self.Dwarf_uint8('address_size'), self.Dwarf_uint8('segment_selector_size'))