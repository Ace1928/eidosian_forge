from Textco BioSoftware, Inc.
from struct import unpack
from Bio.Seq import Seq
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
def _read_pstring(handle):
    """Read a Pascal string.

    A Pascal string is one byte for length followed by the actual string.
    """
    length = _read(handle, 1)
    length = unpack('>B', length)[0]
    data = _read(handle, length).decode('ASCII')
    return data