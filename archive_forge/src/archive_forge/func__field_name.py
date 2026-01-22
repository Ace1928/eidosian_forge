from ..construct import CString
from ..common.utils import struct_parse, elf_assert
from .sections import Section, Symbol
def _field_name(self, name, auxiliary=False):
    """ Return the real field's name of version or a version auxiliary
            entry
        """
    middle = 'a_' if auxiliary else '_'
    return self.field_prefix + middle + name