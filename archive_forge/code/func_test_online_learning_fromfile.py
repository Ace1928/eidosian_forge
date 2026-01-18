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
def test_online_learning_fromfile(self):
    with temporary_file('gensim_fasttext1.tst') as corpus_file, temporary_file('gensim_fasttext2.tst') as new_corpus_file:
        utils.save_as_line_sentence(sentences, corpus_file)
        utils.save_as_line_sentence(new_sentences, new_corpus_file)
        model_hs = FT_gensim(corpus_file=corpus_file, vector_size=12, min_count=1, seed=42, hs=1, negative=0, bucket=BUCKET)
        self.assertTrue(len(model_hs.wv), 12)
        self.assertTrue(model_hs.wv.get_vecattr('graph', 'count'), 3)
        model_hs.build_vocab(corpus_file=new_corpus_file, update=True)
        self.assertEqual(len(model_hs.wv), 14)
        self.assertTrue(model_hs.wv.get_vecattr('graph', 'count'), 4)
        self.assertTrue(model_hs.wv.get_vecattr('artificial', 'count'), 4)