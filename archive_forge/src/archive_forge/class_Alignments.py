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
class Alignments(AlignmentsAbstractBaseClass, list):

    def __init__(self, alignments=()):
        super().__init__(alignments)
        self._index = -1

    def __next__(self):
        index = self._index + 1
        try:
            item = self[index]
        except IndexError:
            raise StopIteration
        self._index = index
        return item

    def rewind(self):
        self._index = -1

    def __len__(self):
        return list.__len__(self)