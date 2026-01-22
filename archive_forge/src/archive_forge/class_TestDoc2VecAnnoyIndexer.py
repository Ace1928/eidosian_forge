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
class TestDoc2VecAnnoyIndexer(unittest.TestCase):

    def setUp(self):
        try:
            import annoy
        except ImportError as e:
            raise unittest.SkipTest('Annoy library is not available: %s' % e)
        from gensim.similarities.annoy import AnnoyIndexer
        self.model = doc2vec.Doc2Vec(SENTENCES, min_count=1)
        self.index = AnnoyIndexer(self.model, 300)
        self.vector = self.model.dv.get_normed_vectors()[0]

    def test_document_is_similar_to_itself(self):
        approx_neighbors = self.index.most_similar(self.vector, 1)
        doc, similarity = approx_neighbors[0]
        self.assertEqual(doc, 0)
        self.assertAlmostEqual(similarity, 1.0, places=2)

    def test_approx_neighbors_match_exact(self):
        approx_neighbors = self.model.dv.most_similar([self.vector], topn=5, indexer=self.index)
        exact_neighbors = self.model.dv.most_similar([self.vector], topn=5)
        approx_words = [neighbor[0] for neighbor in approx_neighbors]
        exact_words = [neighbor[0] for neighbor in exact_neighbors]
        self.assertEqual(approx_words, exact_words)

    def test_save(self):
        fname = get_tmpfile('gensim_similarities.tst.pkl')
        self.index.save(fname)
        self.assertTrue(os.path.exists(fname))
        self.assertTrue(os.path.exists(fname + '.d'))

    def test_load_not_exist(self):
        from gensim.similarities.annoy import AnnoyIndexer
        self.test_index = AnnoyIndexer()
        self.assertRaises(IOError, self.test_index.load, fname='test-index')

    def test_save_load(self):
        from gensim.similarities.annoy import AnnoyIndexer
        fname = get_tmpfile('gensim_similarities.tst.pkl')
        self.index.save(fname)
        self.index2 = AnnoyIndexer()
        self.index2.load(fname)
        self.index2.model = self.model
        self.assertEqual(self.index.index.f, self.index2.index.f)
        self.assertEqual(self.index.labels, self.index2.labels)
        self.assertEqual(self.index.num_trees, self.index2.num_trees)