import logging
import unittest
import math
import os
import numpy
import scipy
from gensim import utils
from gensim.corpora import Dictionary
from gensim.models import word2vec
from gensim.models import doc2vec
from gensim.models import KeyedVectors
from gensim.models import TfidfModel
from gensim import matutils, similarities
from gensim.models import Word2Vec, FastText
from gensim.test.utils import (
from gensim.similarities import UniformTermSimilarityIndex
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import LevenshteinSimilarityIndex
from gensim.similarities.docsim import _nlargest
from gensim.similarities.fastss import editdist
class TestDoc2VecNmslibIndexer(unittest.TestCase):

    def setUp(self):
        try:
            import nmslib
        except ImportError as e:
            raise unittest.SkipTest('NMSLIB library is not available: %s' % e)
        from gensim.similarities.nmslib import NmslibIndexer
        self.model = doc2vec.Doc2Vec(SENTENCES, min_count=1)
        self.index = NmslibIndexer(self.model)
        self.vector = self.model.dv.get_normed_vectors()[0]

    def test_document_is_similar_to_itself(self):
        approx_neighbors = self.index.most_similar(self.vector, 1)
        doc, similarity = approx_neighbors[0]
        self.assertEqual(doc, 0)
        self.assertAlmostEqual(similarity, 1.0, places=2)

    def test_approx_neighbors_match_exact(self):
        approx_neighbors = self.model.dv.most_similar([self.vector], topn=5, indexer=self.index)
        exact_neighbors = self.model.dv.most_similar([self.vector], topn=5)
        approx_tags = [tag for tag, similarity in approx_neighbors]
        exact_tags = [tag for tag, similarity in exact_neighbors]
        self.assertEqual(approx_tags, exact_tags)

    def test_save(self):
        fname = get_tmpfile('gensim_similarities.tst.pkl')
        self.index.save(fname)
        self.assertTrue(os.path.exists(fname))
        self.assertTrue(os.path.exists(fname + '.d'))

    def test_load_not_exist(self):
        from gensim.similarities.nmslib import NmslibIndexer
        self.assertRaises(IOError, NmslibIndexer.load, fname='test-index')

    def test_save_load(self):
        from gensim.similarities.nmslib import NmslibIndexer
        fname = get_tmpfile('gensim_similarities.tst.pkl')
        self.index.save(fname)
        self.index2 = NmslibIndexer.load(fname)
        self.index2.model = self.model
        self.assertEqual(self.index.labels, self.index2.labels)
        self.assertEqual(self.index.index_params, self.index2.index_params)
        self.assertEqual(self.index.query_time_params, self.index2.query_time_params)