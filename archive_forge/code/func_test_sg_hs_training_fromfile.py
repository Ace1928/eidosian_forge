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
@pytest.mark.parametrize('shrink_windows', [True, False])
def test_sg_hs_training_fromfile(shrink_windows):
    with temporary_file('gensim_fasttext.tst') as corpus_file:
        model_gensim = FT_gensim(vector_size=48, sg=1, cbow_mean=1, alpha=0.025, window=5, hs=1, negative=0, min_count=5, epochs=10, batch_words=1000, word_ngrams=1, sample=0.001, min_n=3, max_n=6, sorted_vocab=1, workers=1, min_alpha=0.0, bucket=BUCKET, shrink_windows=shrink_windows)
        lee_data = LineSentence(datapath('lee_background.cor'))
        utils.save_as_line_sentence(lee_data, corpus_file)
        model_gensim.build_vocab(corpus_file=corpus_file)
        orig0 = np.copy(model_gensim.wv.vectors[0])
        model_gensim.train(corpus_file=corpus_file, total_words=model_gensim.corpus_total_words, epochs=model_gensim.epochs)
        assert not (orig0 == model_gensim.wv.vectors[0]).all()
        sims_gensim = model_gensim.wv.most_similar('night', topn=10)
        sims_gensim_words = [word for word, distance in sims_gensim]
        expected_sims_words = [u'night,', u'night.', u'eight', u'nine', u'overnight', u'crew', u'overnight.', u'manslaughter', u'north', u'flight']
        overlaps = set(sims_gensim_words).intersection(expected_sims_words)
        overlap_count = len(overlaps)
        message = f'only {overlap_count} overlap in expected {expected_sims_words} & actual {sims_gensim_words}'
        assert overlap_count >= 2, message