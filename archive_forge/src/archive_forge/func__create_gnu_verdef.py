from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def _create_gnu_verdef(self):
    self.Elf_Verdef = Struct('Elf_Verdef', self.Elf_half('vd_version'), self.Elf_half('vd_flags'), self.Elf_half('vd_ndx'), self.Elf_half('vd_cnt'), self.Elf_word('vd_hash'), self.Elf_word('vd_aux'), self.Elf_word('vd_next'))
    self.Elf_Verdaux = Struct('Elf_Verdaux', self.Elf_word('vda_name'), self.Elf_word('vda_next'))