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
def _get_rows_col(self, coordinates, col, steps, gaps, sequences):
    """Return the alignment contents of multiple rows and one column (PRIVATE).

        This method is called by __getitem__ for invocations of the form

        self[rows, col]

        where rows is a slice object, and col is an integer.
        Return value is a string.
        """
    indices = gaps.cumsum()
    j = indices.searchsorted(col, side='right')
    offset = indices[j] - col
    line = ''
    for sequence, coordinate, step in zip(sequences, coordinates, steps):
        if step[j] == 0:
            line += '-'
        else:
            index = coordinate[j] + step[j] - offset
            line += sequence[index]
    return line