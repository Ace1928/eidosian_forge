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
def test_persistence_fromfile(self):
    """Test storing/loading the entire model."""
    with temporary_file(get_tmpfile('gensim_doc2vec.tst')) as corpus_file:
        save_lee_corpus_as_line_sentence(corpus_file)
        tmpf = get_tmpfile('gensim_doc2vec.tst')
        model = doc2vec.Doc2Vec(corpus_file=corpus_file, min_count=1)
        model.save(tmpf)
        self.models_equal(model, doc2vec.Doc2Vec.load(tmpf))