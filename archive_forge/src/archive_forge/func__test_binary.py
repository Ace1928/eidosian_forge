import itertools
import os
import sys
import tempfile
import unittest
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ...testing import assert_arrays_equal
from ..array_sequence import ArraySequence, concatenate, is_array_sequence
def _test_binary(op, arrseq, scalars, seqs, inplace=False):
    for scalar in scalars:
        orig = arrseq.copy()
        seq = getattr(orig, op)(scalar)
        assert (seq is orig) == inplace
        check_arr_seq(seq, [getattr(e, op)(scalar) for e in arrseq])
    for other in seqs:
        orig = arrseq.copy()
        seq = getattr(orig, op)(other)
        assert seq is not SEQ_DATA['seq']
        check_arr_seq(seq, [getattr(e1, op)(e2) for e1, e2 in zip(arrseq, other)])
    orig = arrseq.copy()
    with pytest.raises(ValueError):
        getattr(orig, op)(orig[::2])
    seq1 = ArraySequence(np.arange(10).reshape(5, 2))
    seq2 = ArraySequence(np.arange(15).reshape(5, 3))
    with pytest.raises(ValueError):
        getattr(seq1, op)(seq2)
    seq1 = ArraySequence(np.arange(12).reshape(2, 2, 3))
    seq2 = ArraySequence(np.arange(8).reshape(2, 2, 2))
    with pytest.raises(ValueError):
        getattr(seq1, op)(seq2)