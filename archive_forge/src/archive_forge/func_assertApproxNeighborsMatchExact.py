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
def assertApproxNeighborsMatchExact(self, model, wv, index):
    vector = wv.get_normed_vectors()[0]
    approx_neighbors = model.most_similar([vector], topn=5, indexer=index)
    exact_neighbors = model.most_similar([vector], topn=5)
    approx_words = [word_id for word_id, similarity in approx_neighbors]
    exact_words = [word_id for word_id, similarity in exact_neighbors]
    self.assertEqual(approx_words, exact_words)