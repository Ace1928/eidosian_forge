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
def _genbank_convert_fasta(in_file, out_file):
    """Fast GenBank to FASTA (PRIVATE)."""
    records = GenBankScanner().parse_records(in_file, do_features=False)
    return SeqIO.write(records, out_file, 'fasta')