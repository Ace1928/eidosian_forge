import logging
import unittest
import numpy as np
import scipy.linalg
from gensim import matutils
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import lsimodel
from gensim.test import basetmtests
from gensim.test.utils import datapath, get_tmpfile
def test_transform_float32(self):
    """Test lsi[vector] transformation."""
    model = lsimodel.LsiModel(self.corpus, num_topics=2, dtype=np.float32)
    u, s, vt = scipy.linalg.svd(matutils.corpus2dense(self.corpus, self.corpus.num_terms), full_matrices=False)
    self.assertTrue(np.allclose(s[:2], model.projection.s))
    self.assertEqual(model.projection.u.dtype, np.float32)
    self.assertEqual(model.projection.s.dtype, np.float32)
    doc = list(self.corpus)[0]
    transformed = model[doc]
    vec = matutils.sparse2full(transformed, 2)
    expected = np.array([-0.6594664, 0.142115444])
    self.assertTrue(np.allclose(abs(vec), abs(expected), atol=1e-05))