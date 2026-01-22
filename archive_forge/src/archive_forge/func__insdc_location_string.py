import warnings
from datetime import datetime
from Bio import BiopythonWarning
from Bio import SeqFeature
from Bio import SeqIO
from Bio.GenBank.Scanner import _ImgtScanner
from Bio.GenBank.Scanner import EmblScanner
from Bio.GenBank.Scanner import GenBankScanner
from Bio.Seq import UndefinedSequenceError
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _insdc_location_string(location, rec_length):
    """Build a GenBank/EMBL location from a (Compound) SimpleLocation (PRIVATE).

    There is a choice of how to show joins on the reverse complement strand,
    GenBank used "complement(join(1,10),(20,100))" while EMBL used to use
    "join(complement(20,100),complement(1,10))" instead (but appears to have
    now adopted the GenBank convention). Notice that the order of the entries
    is reversed! This function therefore uses the first form. In this situation
    we expect the CompoundLocation and its parts to all be marked as
    strand == -1, and to be in the order 19:100 then 0:10.
    """
    try:
        parts = location.parts
        if location.strand == -1:
            return 'complement(%s(%s))' % (location.operator, ','.join((_insdc_location_string_ignoring_strand_and_subfeatures(p, rec_length) for p in parts[::-1])))
        else:
            return '%s(%s)' % (location.operator, ','.join((_insdc_location_string(p, rec_length) for p in parts)))
    except AttributeError:
        loc = _insdc_location_string_ignoring_strand_and_subfeatures(location, rec_length)
        if location.strand == -1:
            return f'complement({loc})'
        else:
            return loc