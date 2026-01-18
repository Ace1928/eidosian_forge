import logging
import unittest
import numbers
from os import remove
import numpy as np
from gensim.corpora import mmcorpus, Dictionary
from gensim.models import atmodel
from gensim import matutils
from gensim.test import basetmtests
from gensim.test.utils import (datapath,
from gensim.matutils import jensen_shannon
def test_update_new_data_new_author(self):
    model = self.class_(corpus, author2doc=author2doc, id2word=dictionary, num_topics=2)
    model.update(corpus_new, author2doc_new)
    sally_topics = model.get_author_topics('sally')
    sally_topics = matutils.sparse2full(sally_topics, model.num_topics)
    self.assertTrue(all(sally_topics > 0))