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
def test_word2vec(self):
    model = word2vec.Word2Vec(TEXTS, min_count=1)
    index = self.indexer(model)
    self.assertVectorIsSimilarToItself(model.wv, index)
    self.assertApproxNeighborsMatchExact(model.wv, model.wv, index)
    self.assertIndexSaved(index)
    self.assertLoadedIndexEqual(index, model)