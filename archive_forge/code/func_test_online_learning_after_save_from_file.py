import logging
import unittest
import os
import bz2
import sys
import tempfile
import subprocess
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import word2vec, keyedvectors
from gensim.utils import check_output
from gensim.test.utils import (
def test_online_learning_after_save_from_file(self):
    """Test that the algorithm is able to add new words to the
        vocabulary and to a trained model when using a sorted vocabulary"""
    with temporary_file(get_tmpfile('gensim_word2vec1.tst')) as corpus_file, temporary_file(get_tmpfile('gensim_word2vec2.tst')) as new_corpus_file:
        utils.save_as_line_sentence(sentences, corpus_file)
        utils.save_as_line_sentence(new_sentences, new_corpus_file)
        tmpf = get_tmpfile('gensim_word2vec.tst')
        model_neg = word2vec.Word2Vec(corpus_file=corpus_file, vector_size=10, min_count=0, seed=42, hs=0, negative=5)
        model_neg.save(tmpf)
        model_neg = word2vec.Word2Vec.load(tmpf)
        self.assertTrue(len(model_neg.wv), 12)
        model_neg.train(corpus_file=corpus_file, total_words=model_neg.corpus_total_words, epochs=model_neg.epochs)
        model_neg.build_vocab(corpus_file=new_corpus_file, update=True)
        model_neg.train(corpus_file=new_corpus_file, total_words=model_neg.corpus_total_words, epochs=model_neg.epochs)
        self.assertEqual(len(model_neg.wv), 14)