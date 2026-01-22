from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def _create_gnu_versym(self):
    self.Elf_Versym = Struct('Elf_Versym', Enum(self.Elf_half('ndx'), **ENUM_VERSYM))