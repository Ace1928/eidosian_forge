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
class TestWordEmbeddingSimilarityIndex(unittest.TestCase):

    def setUp(self):
        self.vectors = KeyedVectors.load_word2vec_format(datapath('euclidean_vectors.bin'), binary=True, datatype=numpy.float64)

    def test_most_similar(self):
        """Test most_similar returns expected results."""
        index = WordEmbeddingSimilarityIndex(self.vectors)
        self.assertLess(0, len(list(index.most_similar(u'holiday', topn=10))))
        self.assertEqual(0, len(list(index.most_similar(u'out-of-dictionary term', topn=10))))
        index = WordEmbeddingSimilarityIndex(self.vectors)
        results = list(index.most_similar(u'holiday', topn=10))
        self.assertLess(0, len(results))
        self.assertGreaterEqual(10, len(results))
        results = list(index.most_similar(u'holiday', topn=20))
        self.assertLess(10, len(results))
        self.assertGreaterEqual(20, len(results))
        index = WordEmbeddingSimilarityIndex(self.vectors)
        terms = [term for term, similarity in index.most_similar(u'holiday', topn=len(self.vectors))]
        self.assertFalse(u'holiday' in terms)
        index = WordEmbeddingSimilarityIndex(self.vectors, threshold=0.0)
        results = list(index.most_similar(u'holiday', topn=10))
        self.assertLess(0, len(results))
        self.assertGreaterEqual(10, len(results))
        index = WordEmbeddingSimilarityIndex(self.vectors, threshold=1.0)
        results = list(index.most_similar(u'holiday', topn=10))
        self.assertEqual(0, len(results))
        index = WordEmbeddingSimilarityIndex(self.vectors, exponent=1.0)
        first_similarities = numpy.array([similarity for term, similarity in index.most_similar(u'holiday', topn=10)])
        index = WordEmbeddingSimilarityIndex(self.vectors, exponent=2.0)
        second_similarities = numpy.array([similarity for term, similarity in index.most_similar(u'holiday', topn=10)])
        self.assertTrue(numpy.allclose(first_similarities ** 2.0, second_similarities))