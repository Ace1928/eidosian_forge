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
class TestLevenshteinSimilarityIndex(unittest.TestCase):

    def setUp(self):
        self.documents = [[u'government', u'denied', u'holiday'], [u'holiday', u'slowing', u'hollingworth']]
        self.dictionary = Dictionary(self.documents)
        max_distance = max((len(term) for term in self.dictionary.values()))
        self.index = LevenshteinSimilarityIndex(self.dictionary, max_distance=max_distance)

    def test_most_similar_topn(self):
        """Test most_similar returns expected results."""
        results = list(self.index.most_similar(u'holiday', topn=0))
        self.assertEqual(0, len(results))
        results = list(self.index.most_similar(u'holiday', topn=1))
        self.assertEqual(1, len(results))
        results = list(self.index.most_similar(u'holiday', topn=4))
        self.assertEqual(4, len(results))
        results = list(self.index.most_similar(u'holiday', topn=len(self.dictionary)))
        self.assertEqual(len(self.dictionary) - 1, len(results))
        self.assertNotIn(u'holiday', results)

    def test_most_similar_result_order(self):
        results = self.index.most_similar(u'holiday', topn=4)
        terms, _ = zip(*results)
        expected_terms = (u'hollingworth', u'denied', u'slowing', u'government')
        self.assertEqual(expected_terms, terms)

    def test_most_similar_alpha(self):
        index = LevenshteinSimilarityIndex(self.dictionary, alpha=1.0)
        first_similarities = numpy.array([similarity for term, similarity in index.most_similar(u'holiday', topn=10)])
        index = LevenshteinSimilarityIndex(self.dictionary, alpha=2.0)
        second_similarities = numpy.array([similarity for term, similarity in index.most_similar(u'holiday', topn=10)])
        self.assertTrue(numpy.allclose(2.0 * first_similarities, second_similarities))

    def test_most_similar_beta(self):
        index = LevenshteinSimilarityIndex(self.dictionary, alpha=1.0, beta=1.0)
        first_similarities = numpy.array([similarity for term, similarity in index.most_similar(u'holiday', topn=10)])
        index = LevenshteinSimilarityIndex(self.dictionary, alpha=1.0, beta=2.0)
        second_similarities = numpy.array([similarity for term, similarity in index.most_similar(u'holiday', topn=10)])
        self.assertTrue(numpy.allclose(first_similarities ** 2.0, second_similarities))