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
def test_online_learning_through_ft_format_saves(self):
    tmpf = get_tmpfile('gensim_ft_format.tst')
    model = FT_gensim(sentences, vector_size=12, min_count=0, seed=42, hs=0, negative=5, bucket=BUCKET)
    gensim.models.fasttext.save_facebook_model(model, tmpf)
    model_reload = gensim.models.fasttext.load_facebook_model(tmpf)
    self.assertTrue(len(model_reload.wv), 12)
    self.assertEqual(len(model_reload.wv), len(model_reload.wv.vectors))
    self.assertEqual(len(model_reload.wv), len(model_reload.wv.vectors_vocab))
    model_reload.build_vocab(new_sentences, update=True)
    model_reload.train(new_sentences, total_examples=model_reload.corpus_count, epochs=model_reload.epochs)
    self.assertEqual(len(model_reload.wv), 14)
    self.assertEqual(len(model_reload.wv), len(model_reload.wv.vectors))
    self.assertEqual(len(model_reload.wv), len(model_reload.wv.vectors_vocab))
    tmpf2 = get_tmpfile('gensim_ft_format2.tst')
    gensim.models.fasttext.save_facebook_model(model_reload, tmpf2)