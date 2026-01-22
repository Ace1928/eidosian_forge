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
def _write_comment(self, record):
    comment = record.annotations['comment']
    if isinstance(comment, str):
        lines = comment.split('\n')
    elif isinstance(comment, (list, tuple)):
        lines = comment
    else:
        raise ValueError('Could not understand comment annotation')
    if not lines:
        return
    for line in lines:
        self._write_multi_line('CC', line)
    self.handle.write('XX\n')