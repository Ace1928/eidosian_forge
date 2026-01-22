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
def assertVectorIsSimilarToItself(self, wv, index):
    vector = wv.get_normed_vectors()[0]
    label = wv.index_to_key[0]
    approx_neighbors = index.most_similar(vector, 1)
    word, similarity = approx_neighbors[0]
    self.assertEqual(word, label)
    self.assertAlmostEqual(similarity, 1.0, places=2)