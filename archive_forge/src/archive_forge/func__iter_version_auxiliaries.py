from ..construct import CString
from ..common.utils import struct_parse, elf_assert
from .sections import Section, Symbol
def _iter_version_auxiliaries(self, entry_offset, count):
    """ Yield all auxiliary entries of a version entry
        """
    name_field = self._field_name('name', auxiliary=True)
    next_field = self._field_name('next', auxiliary=True)
    for _ in range(count):
        entry = struct_parse(self.version_auxiliaries_struct, self.stream, stream_pos=entry_offset)
        name = self.stringtable.get_string(entry[name_field])
        version_aux = VersionAuxiliary(entry, name)
        yield version_aux
        entry_offset += entry[next_field]