from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
class AttributesSubsection(Section):
    """ Subsection of an ELF attributes section.
    """

    def __init__(self, stream, structs, offset, header, subsubsection):
        self.stream = stream
        self.offset = offset
        self.structs = structs
        self.subsubsection = subsubsection
        self.header = struct_parse(header, self.stream, self.offset)
        self.subsubsec_start = self.stream.tell()

    def iter_subsubsections(self, scope=None):
        """ Yield all subsubsections (limit to |scope| if specified).
        """
        for subsubsec in self._make_subsubsections():
            if scope is None or subsubsec.header.tag == scope:
                yield subsubsec

    @property
    def num_subsubsections(self):
        """ Number of subsubsections in the subsection.
        """
        return sum((1 for _ in self.iter_subsubsections()))

    @property
    def subsubsections(self):
        """ List of all subsubsections in the subsection.
        """
        return list(self.iter_subsubsections())

    def _make_subsubsections(self):
        """ Create all subsubsections for this subsection.
        """
        end = self.offset + self['length']
        self.stream.seek(self.subsubsec_start)
        while self.stream.tell() != end:
            subsubsec = self.subsubsection(self.stream, self.structs, self.stream.tell())
            self.stream.seek(self.subsubsec_start + subsubsec.header.value)
            yield subsubsec

    def __getitem__(self, name):
        """ Implement dict-like access to header entries.
        """
        return self.header[name]

    def __repr__(self):
        s = '<%s (%s): %d bytes>'
        return s % (self.__class__.__name__, self.header['vendor_name'], self.header['length'])