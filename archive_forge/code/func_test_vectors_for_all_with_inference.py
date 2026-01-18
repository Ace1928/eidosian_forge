from __future__ import division
import gzip
import io
import logging
import unittest
import os
import shutil
import subprocess
import struct
import sys
import numpy as np
import pytest
from gensim import utils
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText as FT_gensim, FastTextKeyedVectors, _unpack
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import (
from gensim.test.test_word2vec import TestWord2VecModel
import gensim.models._fasttext_bin
from gensim.models.fasttext_inner import compute_ngrams, compute_ngrams_bytes, ft_hash_bytes
import gensim.models.fasttext
def test_vectors_for_all_with_inference(self):
    """Test vectors_for_all can infer new vectors."""
    words = ['responding', 'approached', 'chairman', 'an out-of-vocabulary word', 'another out-of-vocabulary word']
    vectors_for_all = self.test_model.wv.vectors_for_all(words)
    expected = 5
    predicted = len(vectors_for_all)
    assert expected == predicted
    expected = self.test_model.wv['responding']
    predicted = vectors_for_all['responding']
    assert np.allclose(expected, predicted)
    smaller_distance = np.linalg.norm(vectors_for_all['an out-of-vocabulary word'] - vectors_for_all['another out-of-vocabulary word'])
    greater_distance = np.linalg.norm(vectors_for_all['an out-of-vocabulary word'] - vectors_for_all['responding'])
    assert greater_distance > smaller_distance