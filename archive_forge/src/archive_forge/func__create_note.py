from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def _create_note(self, e_type=None):
    self.Elf_ugid = self.Elf_half if self.elfclass == 32 and self.e_machine in {'EM_MN10300', 'EM_ARM', 'EM_CRIS', 'EM_CYGNUS_FRV', 'EM_386', 'EM_M32R', 'EM_68K', 'EM_S390', 'EM_SH', 'EM_SPARC'} else self.Elf_word
    self.Elf_Nhdr = Struct('Elf_Nhdr', self.Elf_word('n_namesz'), self.Elf_word('n_descsz'), Enum(self.Elf_word('n_type'), **ENUM_NOTE_N_TYPE if e_type != 'ET_CORE' else ENUM_CORE_NOTE_N_TYPE))
    if self.elfclass == 32:
        self.Elf_Prpsinfo = Struct('Elf_Prpsinfo', self.Elf_byte('pr_state'), String('pr_sname', 1), self.Elf_byte('pr_zomb'), self.Elf_byte('pr_nice'), self.Elf_xword('pr_flag'), self.Elf_ugid('pr_uid'), self.Elf_ugid('pr_gid'), self.Elf_word('pr_pid'), self.Elf_word('pr_ppid'), self.Elf_word('pr_pgrp'), self.Elf_word('pr_sid'), String('pr_fname', 16), String('pr_psargs', 80))
    else:
        self.Elf_Prpsinfo = Struct('Elf_Prpsinfo', self.Elf_byte('pr_state'), String('pr_sname', 1), self.Elf_byte('pr_zomb'), self.Elf_byte('pr_nice'), Padding(4), self.Elf_xword('pr_flag'), self.Elf_ugid('pr_uid'), self.Elf_ugid('pr_gid'), self.Elf_word('pr_pid'), self.Elf_word('pr_ppid'), self.Elf_word('pr_pgrp'), self.Elf_word('pr_sid'), String('pr_fname', 16), String('pr_psargs', 80))
    self.Elf_Nt_File = Struct('Elf_Nt_File', self.Elf_xword('num_map_entries'), self.Elf_xword('page_size'), Array(lambda ctx: ctx.num_map_entries, Struct('Elf_Nt_File_Entry', self.Elf_addr('vm_start'), self.Elf_addr('vm_end'), self.Elf_offset('page_offset'))), Array(lambda ctx: ctx.num_map_entries, CString('filename')))