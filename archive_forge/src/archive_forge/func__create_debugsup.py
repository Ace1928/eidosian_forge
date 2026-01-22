from logging.config import valid_ident
from ..construct import (
from ..common.construct_utils import (RepeatUntilExcluding, ULEB128, SLEB128,
from .enums import *
def _create_debugsup(self):
    self.Dwarf_debugsup = Struct('Elf_debugsup', self.Dwarf_int16('version'), self.Dwarf_uint8('is_supplementary'), CString('sup_filename'))