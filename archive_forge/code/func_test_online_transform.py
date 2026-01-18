import logging
import unittest
import numpy as np
import scipy.linalg
from gensim import matutils
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import lsimodel
from gensim.test import basetmtests
from gensim.test.utils import datapath, get_tmpfile
def test_online_transform(self):
    corpus = list(self.corpus)
    doc = corpus[0]
    model2 = lsimodel.LsiModel(corpus=corpus, num_topics=5)
    model = lsimodel.LsiModel(corpus=None, id2word=model2.id2word, num_topics=5)
    model.add_documents([corpus[0]])
    transformed = model[doc]
    vec = matutils.sparse2full(transformed, model.num_topics)
    expected = np.array([-1.73205078, 0.0, 0.0, 0.0, 0.0])
    self.assertTrue(np.allclose(abs(vec), abs(expected), atol=1e-06))
    model.add_documents(corpus[1:5], chunksize=2)
    transformed = model[doc]
    vec = matutils.sparse2full(transformed, model.num_topics)
    expected = np.array([-0.66493785, -0.28314203, -1.56376302, 0.05488682, 0.17123269])
    self.assertTrue(np.allclose(abs(vec), abs(expected), atol=1e-06))
    model.add_documents(corpus[5:])
    vec1 = matutils.sparse2full(model[doc], model.num_topics)
    vec2 = matutils.sparse2full(model2[doc], model2.num_topics)
    self.assertTrue(np.allclose(abs(vec1), abs(vec2), atol=1e-05))