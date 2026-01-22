from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def _create_chdr(self):
    fields = [Enum(self.Elf_word('ch_type'), **ENUM_ELFCOMPRESS_TYPE), self.Elf_xword('ch_size'), self.Elf_xword('ch_addralign')]
    if self.elfclass == 64:
        fields.insert(1, self.Elf_word('ch_reserved'))
    self.Elf_Chdr = Struct('Elf_Chdr', *fields)