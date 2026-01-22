from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def _create_attributes_subsection(self):
    self.Elf_Attr_Subsection_Header = Struct('Elf_Attr_Subsection', self.Elf_word('length'), self.Elf_ntbs('vendor_name', encoding='utf-8'))