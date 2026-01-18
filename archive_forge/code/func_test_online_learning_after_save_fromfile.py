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
def test_online_learning_after_save_fromfile(self):
    with temporary_file('gensim_fasttext1.tst') as corpus_file, temporary_file('gensim_fasttext2.tst') as new_corpus_file:
        utils.save_as_line_sentence(sentences, corpus_file)
        utils.save_as_line_sentence(new_sentences, new_corpus_file)
        tmpf = get_tmpfile('gensim_fasttext.tst')
        model_neg = FT_gensim(corpus_file=corpus_file, vector_size=12, min_count=0, seed=42, hs=0, negative=5, bucket=BUCKET)
        model_neg.save(tmpf)
        model_neg = FT_gensim.load(tmpf)
        self.assertTrue(len(model_neg.wv), 12)
        model_neg.build_vocab(corpus_file=new_corpus_file, update=True)
        model_neg.train(corpus_file=new_corpus_file, total_words=model_neg.corpus_total_words, epochs=model_neg.epochs)
        self.assertEqual(len(model_neg.wv), 14)