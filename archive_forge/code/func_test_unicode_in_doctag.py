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
def test_unicode_in_doctag(self):
    """Test storing document vectors of a model with unicode titles."""
    model = doc2vec.Doc2Vec(DocsLeeCorpus(unicode_tags=True), min_count=1)
    tmpf = get_tmpfile('gensim_doc2vec.tst')
    try:
        model.save_word2vec_format(tmpf, doctag_vec=True, word_vec=True, binary=True)
    except UnicodeEncodeError:
        self.fail('Failed storing unicode title.')