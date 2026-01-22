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
class TestSoftCosineSimilarity(_TestSimilarityABC):

    def setUp(self):
        self.cls = similarities.SoftCosineSimilarity
        self.tfidf = TfidfModel(dictionary=DICTIONARY)
        similarity_matrix = scipy.sparse.identity(12, format='lil')
        similarity_matrix[DICTIONARY.token2id['user'], DICTIONARY.token2id['human']] = 0.5
        similarity_matrix[DICTIONARY.token2id['human'], DICTIONARY.token2id['user']] = 0.5
        self.similarity_matrix = SparseTermSimilarityMatrix(similarity_matrix)

    def factoryMethod(self):
        return self.cls(CORPUS, self.similarity_matrix)

    def test_full(self, num_best=None):
        index = self.cls(CORPUS, self.similarity_matrix, num_best=num_best)
        query = DICTIONARY.doc2bow(TEXTS[0])
        sims = index[query]
        if num_best is not None:
            for i, sim in sims:
                self.assertTrue(numpy.alltrue(sim <= 1.0))
                self.assertTrue(numpy.alltrue(sim >= 0.0))
        else:
            self.assertAlmostEqual(1.0, sims[0])
            self.assertTrue(numpy.alltrue(sims[1:] >= 0.0))
            self.assertTrue(numpy.alltrue(sims[1:] < 1.0))
        for query in (CORPUS, self.tfidf[CORPUS]):
            index = self.cls(query, self.similarity_matrix, num_best=num_best)
            sims = index[query]
            if num_best is not None:
                for result in sims:
                    for i, sim in result:
                        self.assertTrue(numpy.alltrue(sim <= 1.0))
                        self.assertTrue(numpy.alltrue(sim >= 0.0))
            else:
                for i, result in enumerate(sims):
                    self.assertAlmostEqual(1.0, result[i])
                    self.assertTrue(numpy.alltrue(result[:i] >= 0.0))
                    self.assertTrue(numpy.alltrue(result[:i] < 1.0))
                    self.assertTrue(numpy.alltrue(result[i + 1:] >= 0.0))
                    self.assertTrue(numpy.alltrue(result[i + 1:] < 1.0))

    def test_non_increasing(self):
        """ Check that similarities are non-increasing when `num_best` is not `None`."""
        index = self.cls(CORPUS, self.similarity_matrix, num_best=5)
        query = DICTIONARY.doc2bow(TEXTS[0])
        sims = index[query]
        sims2 = numpy.asarray(sims)[:, 1]
        cond = sum(numpy.diff(sims2) <= 0) == len(sims2) - 1
        self.assertTrue(cond)

    def test_chunking(self):
        index = self.cls(CORPUS, self.similarity_matrix)
        query = [DICTIONARY.doc2bow(document) for document in TEXTS[:3]]
        sims = index[query]
        for i in range(3):
            self.assertTrue(numpy.alltrue(sims[i, i] == 1.0))
        index.num_best = 5
        sims = index[query]
        for i, chunk in enumerate(sims):
            expected = i
            self.assertAlmostEqual(expected, chunk[0][0], places=2)
            expected = 1.0
            self.assertAlmostEqual(expected, chunk[0][1], places=2)

    def test_iter(self):
        index = self.cls(CORPUS, self.similarity_matrix)
        for sims in index:
            self.assertTrue(numpy.alltrue(sims >= 0.0))
            self.assertTrue(numpy.alltrue(sims <= 1.0))