from collections import namedtuple
from ..common.exceptions import ELFRelocationError
from ..common.utils import elf_assert, struct_parse
from .sections import Section
from .enums import (
from ..construct import Container
class RelocationSection(Section, RelocationTable):
    """ ELF relocation section. Serves as a collection of Relocation entries.
    """

    def __init__(self, header, name, elffile):
        Section.__init__(self, header, name, elffile)
        RelocationTable.__init__(self, self.elffile, self['sh_offset'], self['sh_size'], header['sh_type'] == 'SHT_RELA')
        elf_assert(header['sh_type'] in ('SHT_REL', 'SHT_RELA'), 'Unknown relocation type section')
        elf_assert(header['sh_entsize'] == self.entry_size, 'Expected sh_entsize of %s section to be %s' % (header['sh_type'], self.entry_size))