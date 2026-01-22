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
def _read_overhang(handle):
    """Read an overhang specification.

    An overhang is represented in a XDNA file as:
      - a Pascal string containing the text representation of the overhang
        length, which also indicates the nature of the overhang:
        - a length of zero means no overhang,
        - a negative length means a 3' overhang,
        - a positive length means a 5' overhang;
      - the actual overhang sequence.

    Examples:
      - 0x01 0x30: no overhang ("0", as a P-string)
      - 0x01 0x32 0x41 0x41: 5' AA overhang (P-string "2", then "AA")
      - 0x02 0x2D 0x31 0x43: 3' C overhang (P-string "-1", then "C")

    Returns a tuple (length, sequence).

    """
    length = _read_pstring_as_integer(handle)
    if length != 0:
        overhang = _read(handle, abs(length))
        return (length, overhang)
    else:
        return (None, None)