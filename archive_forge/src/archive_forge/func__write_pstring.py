import warnings
from re import match
from struct import pack
from struct import unpack
from Bio import BiopythonWarning
from Bio.Seq import Seq
from Bio.SeqFeature import ExactPosition
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _write_pstring(self, s):
    """Write the given string as a Pascal string."""
    if len(s) > 255:
        self._has_truncated_strings = True
        s = s[:255]
    self.handle.write(pack('>B', len(s)))
    self.handle.write(s.encode('ASCII'))