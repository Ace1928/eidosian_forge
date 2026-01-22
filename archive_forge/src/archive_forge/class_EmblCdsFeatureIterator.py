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
class EmblCdsFeatureIterator(SequenceIterator):
    """Parser for EMBL files, creating a SeqRecord for each CDS feature."""

    def __init__(self, source):
        """Break up a EMBL file into SeqRecord objects for each CDS feature.

        Argument source is a file-like object opened in text mode or a path to a file.

        Every section from the LOCUS line to the terminating // can contain
        many CDS features.  These are returned as with the stated amino acid
        translation sequence (if given).
        """
        super().__init__(source, mode='t', fmt='EMBL')

    def parse(self, handle):
        """Start parsing the file, and return a SeqRecord generator."""
        return EmblScanner(debug=0).parse_cds_features(handle)