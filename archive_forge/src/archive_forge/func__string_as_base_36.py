import re
import struct
from Bio import StreamModeError
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _string_as_base_36(string):
    """Interpret a string as a base-36 number as per 454 manual (PRIVATE)."""
    total = 0
    for c, power in zip(string[::-1], _powers_of_36):
        if 48 <= ord(c) <= 57:
            val = ord(c) - 22
        elif 65 <= ord(c) <= 90:
            val = ord(c) - 65
        elif 97 <= ord(c) <= 122:
            val = ord(c) - 97
        else:
            val = 0
        total += val * power
    return total