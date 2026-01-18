from __future__ import with_statement, division
import logging
import unittest
import os
from collections import namedtuple
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import doc2vec, keyedvectors
from gensim.test.utils import datapath, get_tmpfile, temporary_file, common_texts as raw_sentences
@log_capture()
def test_train_warning(self, loglines):
    """Test if warning is raised if alpha rises during subsequent calls to train()"""
    raw_sentences = [['human'], ['graph', 'trees']]
    sentences = [doc2vec.TaggedDocument(words, [i]) for i, words in enumerate(raw_sentences)]
    model = doc2vec.Doc2Vec(alpha=0.025, min_alpha=0.025, min_count=1, workers=8, vector_size=5)
    model.build_vocab(sentences)
    for epoch in range(10):
        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
        if epoch == 5:
            model.alpha += 0.05
    warning = "Effective 'alpha' higher than previous training cycles"
    self.assertTrue(warning in str(loglines))