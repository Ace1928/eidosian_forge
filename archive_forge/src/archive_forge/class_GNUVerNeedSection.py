from ..construct import CString
from ..common.utils import struct_parse, elf_assert
from .sections import Section, Symbol
class GNUVerNeedSection(GNUVersionSection):
    """ ELF SUNW or GNU Version Needed table section.
        Has an associated StringTableSection that's passed in the constructor.
    """

    def __init__(self, header, name, elffile, stringtable):
        super(GNUVerNeedSection, self).__init__(header, name, elffile, stringtable, 'vn', elffile.structs.Elf_Verneed, elffile.structs.Elf_Vernaux)
        self._has_indexes = None

    def has_indexes(self):
        """ Return True if at least one version definition entry has an index
            that is stored in the vna_other field.
            This information is used for symbol versioning
        """
        if self._has_indexes is None:
            self._has_indexes = False
            for _, vernaux_iter in self.iter_versions():
                for vernaux in vernaux_iter:
                    if vernaux['vna_other']:
                        self._has_indexes = True
                        break
        return self._has_indexes

    def iter_versions(self):
        for verneed, vernaux in super(GNUVerNeedSection, self).iter_versions():
            verneed.name = self.stringtable.get_string(verneed['vn_file'])
            yield (verneed, vernaux)

    def get_version(self, index):
        """ Get the version information located at index #n in the table
            Return boths the verneed structure and the vernaux structure
            that contains the name of the version
        """
        for verneed, vernaux_iter in self.iter_versions():
            for vernaux in vernaux_iter:
                if vernaux['vna_other'] == index:
                    return (verneed, vernaux)
        return None