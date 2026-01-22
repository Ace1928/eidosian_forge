import sys
import collections
import copy
import importlib
import types
import warnings
import numbers
from itertools import zip_longest
from abc import ABC, abstractmethod
from typing import Dict
from Bio.Align import _pairwisealigner  # type: ignore
from Bio.Align import _codonaligner  # type: ignore
from Bio.Align import substitution_matrices
from Bio.Data import CodonTable
from Bio.Seq import Seq, MutableSeq, reverse_complement, UndefinedSequenceError
from Bio.Seq import translate
from Bio.SeqRecord import SeqRecord, _RestrictedDict
def _str_line(self, record, length=50):
    """Return a truncated string representation of a SeqRecord (PRIVATE).

        This is a PRIVATE function used by the __str__ method.
        """
    if record.seq.__class__.__name__ == 'CodonSeq':
        if len(record.seq) <= length:
            return f'{record.seq} {record.id}'
        else:
            return '%s...%s %s' % (record.seq[:length - 3], record.seq[-3:], record.id)
    elif len(record.seq) <= length:
        return f'{record.seq} {record.id}'
    else:
        return '%s...%s %s' % (record.seq[:length - 6], record.seq[-3:], record.id)