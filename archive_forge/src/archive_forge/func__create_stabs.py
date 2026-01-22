from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def _create_stabs(self):
    self.Elf_Stabs = Struct('Elf_Stabs', self.Elf_word('n_strx'), self.Elf_byte('n_type'), self.Elf_byte('n_other'), self.Elf_half('n_desc'), self.Elf_word('n_value'))