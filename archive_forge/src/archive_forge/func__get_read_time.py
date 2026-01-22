import re
import struct
from Bio import StreamModeError
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _get_read_time(read_name):
    """Extract time from first 6 characters of read name (PRIVATE)."""
    time_list = []
    remainder = _string_as_base_36(read_name[:6])
    for denominator in _time_denominators:
        this_term, remainder = divmod(remainder, denominator)
        time_list.append(this_term)
    time_list.append(remainder)
    time_list[0] += 2000
    return time_list