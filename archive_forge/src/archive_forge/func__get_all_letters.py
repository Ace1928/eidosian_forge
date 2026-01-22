import math
import sys
import warnings
from collections import Counter
from Bio import BiopythonDeprecationWarning
from Bio.Seq import Seq
def _get_all_letters(self):
    """Return a string containing the expected letters in the alignment (PRIVATE)."""
    set_letters = set()
    for record in self.alignment:
        set_letters.update(record.seq)
    list_letters = sorted(set_letters)
    all_letters = ''.join(list_letters)
    return all_letters