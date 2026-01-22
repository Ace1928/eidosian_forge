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
@staticmethod
def _split_multi_line(text, max_len):
    """Return a list of strings (PRIVATE).

        Any single words which are too long get returned as a whole line
        (e.g. URLs) without an exception or warning.
        """
    text = text.strip()
    if len(text) <= max_len:
        return [text]
    words = text.split()
    text = ''
    while words and len(text) + 1 + len(words[0]) <= max_len:
        text += ' ' + words.pop(0)
        text = text.strip()
    answer = [text]
    while words:
        text = words.pop(0)
        while words and len(text) + 1 + len(words[0]) <= max_len:
            text += ' ' + words.pop(0)
            text = text.strip()
        answer.append(text)
    assert not words
    return answer