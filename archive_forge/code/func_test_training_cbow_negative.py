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
def test_training_cbow_negative(self):
    """Test CBOW (negative sampling) word2vec training."""
    model = word2vec.Word2Vec(vector_size=2, min_count=1, sg=0, hs=0, negative=2)
    model.build_vocab(sentences)
    self.assertTrue(model.wv.vectors.shape == (len(model.wv), 2))
    self.assertTrue(model.syn1neg.shape == (len(model.wv), 2))
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    sims = model.wv.most_similar('graph', topn=10)
    graph_vector = model.wv.get_vector('graph', norm=True)
    sims2 = model.wv.most_similar(positive=[graph_vector], topn=11)
    sims2 = [(w, sim) for w, sim in sims2 if w != 'graph']
    self.assertEqual(sims, sims2)
    model2 = word2vec.Word2Vec(sentences, vector_size=2, min_count=1, sg=0, hs=0, negative=2)
    self.models_equal(model, model2)