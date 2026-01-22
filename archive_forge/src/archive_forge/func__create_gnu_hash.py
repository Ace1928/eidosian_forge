from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def _create_gnu_hash(self):
    self.Gnu_Hash = Struct('Gnu_Hash', self.Elf_word('nbuckets'), self.Elf_word('symoffset'), self.Elf_word('bloom_size'), self.Elf_word('bloom_shift'), Array(lambda ctx: ctx['bloom_size'], self.Elf_xword('bloom')), Array(lambda ctx: ctx['nbuckets'], self.Elf_word('buckets')))