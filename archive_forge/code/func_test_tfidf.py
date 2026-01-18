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
def test_tfidf(self):
    """Test the tfidf parameter of the matrix constructor."""
    matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, nonzero_limit=1).matrix.todense()
    expected_matrix = numpy.array([[1.0, 0.5, 0.0, 0.0, 0.0], [0.5, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
    self.assertTrue(numpy.all(expected_matrix == matrix))
    matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, nonzero_limit=1, tfidf=self.tfidf).matrix.todense()
    expected_matrix = numpy.array([[1.0, 0.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.5, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
    self.assertTrue(numpy.all(expected_matrix == matrix))