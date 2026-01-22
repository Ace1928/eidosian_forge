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
class EmblIterator(SequenceIterator):
    """Parser for EMBL files."""

    def __init__(self, source):
        """Break up an EMBL file into SeqRecord objects.

        Argument source is a file-like object opened in text mode or a path to a file.
        Every section from the LOCUS line to the terminating // becomes
        a single SeqRecord with associated annotation and features.

        Note that for genomes or chromosomes, there is typically only
        one record.

        This gets called internally by Bio.SeqIO for the EMBL file format:

        >>> from Bio import SeqIO
        >>> for record in SeqIO.parse("EMBL/epo_prt_selection.embl", "embl"):
        ...     print(record.id)
        ...
        A00022.1
        A00028.1
        A00031.1
        A00034.1
        A00060.1
        A00071.1
        A00072.1
        A00078.1
        CQ797900.1

        Equivalently,

        >>> with open("EMBL/epo_prt_selection.embl") as handle:
        ...     for record in EmblIterator(handle):
        ...         print(record.id)
        ...
        A00022.1
        A00028.1
        A00031.1
        A00034.1
        A00060.1
        A00071.1
        A00072.1
        A00078.1
        CQ797900.1

        """
        super().__init__(source, mode='t', fmt='EMBL')

    def parse(self, handle):
        """Start parsing the file, and return a SeqRecord generator."""
        records = EmblScanner(debug=0).parse_records(handle)
        return records