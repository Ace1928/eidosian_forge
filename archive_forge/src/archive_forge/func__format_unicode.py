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
def _format_unicode(self):
    """Return default string representation (PRIVATE).

        Helper for self.format().
        """
    seqs = []
    names = []
    coordinates = self.coordinates.copy()
    for seq, row in zip(self.sequences, coordinates):
        seq = self._convert_sequence_string(seq)
        if seq is None:
            return self._format_generalized()
        if row[0] > row[-1]:
            row[:] = len(seq) - row[:]
            seq = reverse_complement(seq)
        seqs.append(seq)
        try:
            name = seq.id
        except AttributeError:
            if len(self.sequences) == 2:
                if len(names) == 0:
                    name = 'target'
                else:
                    name = 'query'
            else:
                name = ''
        else:
            name = name[:9]
        name = name.ljust(10)
        names.append(name)
    steps = np.diff(coordinates, 1).max(0)
    aligned_seqs = []
    for row, seq in zip(coordinates, seqs):
        aligned_seq = ''
        start = row[0]
        for step, end in zip(steps, row[1:]):
            if end == start:
                aligned_seq += '-' * step
            else:
                aligned_seq += seq[start:end]
            start = end
        aligned_seqs.append(aligned_seq)
    if len(seqs) > 2:
        return '\n'.join(aligned_seqs) + '\n'
    aligned_seq1, aligned_seq2 = aligned_seqs
    pattern = ''
    for c1, c2 in zip(aligned_seq1, aligned_seq2):
        if c1 == c2:
            c = '|'
        elif c1 == '-' or c2 == '-':
            c = '-'
        else:
            c = '.'
        pattern += c
    return f'{aligned_seq1}\n{pattern}\n{aligned_seq2}\n'