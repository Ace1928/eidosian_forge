from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def _create_shdr(self):
    """Section header parsing.

        Depends on e_machine because of machine-specific values in sh_type.
        """
    sh_type_dict = ENUM_SH_TYPE_BASE
    if self.e_machine == 'EM_ARM':
        sh_type_dict = ENUM_SH_TYPE_ARM
    elif self.e_machine == 'EM_X86_64':
        sh_type_dict = ENUM_SH_TYPE_AMD64
    elif self.e_machine == 'EM_MIPS':
        sh_type_dict = ENUM_SH_TYPE_MIPS
    if self.e_machine == 'EM_RISCV':
        sh_type_dict = ENUM_SH_TYPE_RISCV
    self.Elf_Shdr = Struct('Elf_Shdr', self.Elf_word('sh_name'), Enum(self.Elf_word('sh_type'), **sh_type_dict), self.Elf_xword('sh_flags'), self.Elf_addr('sh_addr'), self.Elf_offset('sh_offset'), self.Elf_xword('sh_size'), self.Elf_word('sh_link'), self.Elf_word('sh_info'), self.Elf_xword('sh_addralign'), self.Elf_xword('sh_entsize'))