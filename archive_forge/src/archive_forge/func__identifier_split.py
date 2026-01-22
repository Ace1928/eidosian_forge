import re
from typing import List
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import AlignmentIterator
from .Interfaces import SequentialAlignmentWriter
def _identifier_split(identifier):
    """Return (name, start, end) string tuple from an identifier (PRIVATE)."""
    id, loc, strand = identifier.split(':')
    start, end = map(int, loc.split('-'))
    start -= 1
    return (id, start, end, strand)