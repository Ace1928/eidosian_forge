from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def _create_elf_hash(self):
    self.Elf_Hash = Struct('Elf_Hash', self.Elf_word('nbuckets'), self.Elf_word('nchains'), Array(lambda ctx: ctx['nbuckets'], self.Elf_word('buckets')), Array(lambda ctx: ctx['nchains'], self.Elf_word('chains')))