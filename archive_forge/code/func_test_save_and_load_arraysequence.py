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
def test_save_and_load_arraysequence(self):
    with tempfile.TemporaryFile(mode='w+b', suffix='.npz') as f:
        seq = ArraySequence()
        seq.save(f)
        f.seek(0, os.SEEK_SET)
        loaded_seq = ArraySequence.load(f)
        assert_array_equal(loaded_seq._data, seq._data)
        assert_array_equal(loaded_seq._offsets, seq._offsets)
        assert_array_equal(loaded_seq._lengths, seq._lengths)
    with tempfile.TemporaryFile(mode='w+b', suffix='.npz') as f:
        seq = SEQ_DATA['seq']
        seq.save(f)
        f.seek(0, os.SEEK_SET)
        loaded_seq = ArraySequence.load(f)
        assert_array_equal(loaded_seq._data, seq._data)
        assert_array_equal(loaded_seq._offsets, seq._offsets)
        assert_array_equal(loaded_seq._lengths, seq._lengths)
        loaded_seq.append(SEQ_DATA['data'][0])