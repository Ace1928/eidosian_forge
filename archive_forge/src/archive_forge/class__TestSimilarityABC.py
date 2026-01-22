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
@unittest.skip('skipping abstract base class')
class _TestSimilarityABC(unittest.TestCase):
    """
    Base class for SparseMatrixSimilarity and MatrixSimilarity unit tests.
    """

    def factoryMethod(self):
        """Creates a SimilarityABC instance."""
        return self.cls(CORPUS, num_features=len(DICTIONARY))

    def test_full(self, num_best=None, shardsize=100):
        if self.cls == similarities.Similarity:
            index = self.cls(None, CORPUS, num_features=len(DICTIONARY), shardsize=shardsize)
        else:
            index = self.cls(CORPUS, num_features=len(DICTIONARY))
        if isinstance(index, similarities.MatrixSimilarity):
            expected = numpy.array([[0.57735026, 0.57735026, 0.57735026, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.40824831, 0.0, 0.40824831, 0.40824831, 0.40824831, 0.40824831, 0.40824831, 0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.40824831, 0.0, 0.0, 0.0, 0.81649661, 0.0, 0.40824831, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.57735026, 0.57735026, 0.0, 0.0, 0.57735026, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.70710677, 0.70710677, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.57735026, 0.57735026, 0.57735026], [0.0, 0.0, 0.0, 0.0, 0.0, 0.57735026, 0.0, 0.0, 0.0, 0.0, 0.57735026, 0.57735026]], dtype=numpy.float32)
            self.assertTrue(numpy.allclose(sorted(expected.flat), sorted(index.index.flat)))
        index.num_best = num_best
        query = CORPUS[0]
        sims = index[query]
        expected = [(0, 0.99999994), (2, 0.28867513), (3, 0.23570226), (1, 0.23570226)][:num_best]
        expected = matutils.sparse2full(expected, len(index))
        if num_best is not None:
            sims = matutils.sparse2full(sims, len(index))
        self.assertTrue(numpy.allclose(expected, sims))
        if self.cls == similarities.Similarity:
            index.destroy()

    def test_num_best(self):
        if self.cls == similarities.WmdSimilarity and (not POT_EXT):
            self.skipTest('POT not installed')
        for num_best in [None, 0, 1, 9, 1000]:
            self.testFull(num_best=num_best)

    def test_full2sparse_clipped(self):
        vec = [0.8, 0.2, 0.0, 0.0, -0.1, -0.15]
        expected = [(0, 0.8), (1, 0.2), (5, -0.15)]
        self.assertTrue(matutils.full2sparse_clipped(vec, topn=3), expected)

    def test_scipy2scipy_clipped(self):
        vec = [0.8, 0.2, 0.0, 0.0, -0.1, -0.15]
        expected = [(0, 0.8), (1, 0.2), (5, -0.15)]
        vec_scipy = scipy.sparse.csr_matrix(vec)
        vec_scipy_clipped = matutils.scipy2scipy_clipped(vec_scipy, topn=3)
        self.assertTrue(scipy.sparse.issparse(vec_scipy_clipped))
        self.assertTrue(matutils.scipy2sparse(vec_scipy_clipped), expected)
        vec = [0.8, 0.2, 0.0, 0.0, -0.1, -0.15]
        expected = [(0, 0.8), (1, 0.2), (5, -0.15)]
        matrix_scipy = scipy.sparse.csr_matrix([vec] * 3)
        matrix_scipy_clipped = matutils.scipy2scipy_clipped(matrix_scipy, topn=3)
        self.assertTrue(scipy.sparse.issparse(matrix_scipy_clipped))
        self.assertTrue([matutils.scipy2sparse(x) for x in matrix_scipy_clipped], [expected] * 3)

    def test_empty_query(self):
        index = self.factoryMethod()
        if isinstance(index, similarities.WmdSimilarity) and (not POT_EXT):
            self.skipTest('POT not installed')
        query = []
        try:
            sims = index[query]
            self.assertTrue(sims is not None)
        except IndexError:
            self.assertTrue(False)

    def test_chunking(self):
        if self.cls == similarities.Similarity:
            index = self.cls(None, CORPUS, num_features=len(DICTIONARY), shardsize=5)
        else:
            index = self.cls(CORPUS, num_features=len(DICTIONARY))
        query = CORPUS[:3]
        sims = index[query]
        expected = numpy.array([[0.99999994, 0.23570226, 0.28867513, 0.23570226, 0.0, 0.0, 0.0, 0.0, 0.0], [0.23570226, 1.0, 0.40824831, 0.33333334, 0.70710677, 0.0, 0.0, 0.0, 0.23570226], [0.28867513, 0.40824831, 1.0, 0.61237246, 0.28867513, 0.0, 0.0, 0.0, 0.0]], dtype=numpy.float32)
        self.assertTrue(numpy.allclose(expected, sims))
        index.num_best = 3
        sims = index[query]
        expected = [[(0, 0.99999994), (2, 0.28867513), (1, 0.23570226)], [(1, 1.0), (4, 0.70710677), (2, 0.40824831)], [(2, 1.0), (3, 0.61237246), (1, 0.40824831)]]
        self.assertTrue(numpy.allclose(expected, sims))
        if self.cls == similarities.Similarity:
            index.destroy()

    def test_iter(self):
        if self.cls == similarities.Similarity:
            index = self.cls(None, CORPUS, num_features=len(DICTIONARY), shardsize=5)
        else:
            index = self.cls(CORPUS, num_features=len(DICTIONARY))
        sims = [sim for sim in index]
        expected = numpy.array([[0.99999994, 0.23570226, 0.28867513, 0.23570226, 0.0, 0.0, 0.0, 0.0, 0.0], [0.23570226, 1.0, 0.40824831, 0.33333334, 0.70710677, 0.0, 0.0, 0.0, 0.23570226], [0.28867513, 0.40824831, 1.0, 0.61237246, 0.28867513, 0.0, 0.0, 0.0, 0.0], [0.23570226, 0.33333334, 0.61237246, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.70710677, 0.28867513, 0.0, 0.99999994, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.70710677, 0.57735026, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.70710677, 0.99999994, 0.81649655, 0.40824828], [0.0, 0.0, 0.0, 0.0, 0.0, 0.57735026, 0.81649655, 0.99999994, 0.66666663], [0.0, 0.23570226, 0.0, 0.0, 0.0, 0.0, 0.40824828, 0.66666663, 0.99999994]], dtype=numpy.float32)
        self.assertTrue(numpy.allclose(expected, sims))
        if self.cls == similarities.Similarity:
            index.destroy()

    def test_persistency(self):
        if self.cls == similarities.WmdSimilarity and (not POT_EXT):
            self.skipTest('POT not installed')
        fname = get_tmpfile('gensim_similarities.tst.pkl')
        index = self.factoryMethod()
        index.save(fname)
        index2 = self.cls.load(fname)
        if self.cls == similarities.Similarity:
            self.assertTrue(len(index.shards) == len(index2.shards))
            index.destroy()
        else:
            if isinstance(index, similarities.SparseMatrixSimilarity):
                index.index = index.index.todense()
                index2.index = index2.index.todense()
            self.assertTrue(numpy.allclose(index.index, index2.index))
            self.assertEqual(index.num_best, index2.num_best)

    def test_persistency_compressed(self):
        if self.cls == similarities.WmdSimilarity and (not POT_EXT):
            self.skipTest('POT not installed')
        fname = get_tmpfile('gensim_similarities.tst.pkl.gz')
        index = self.factoryMethod()
        index.save(fname)
        index2 = self.cls.load(fname)
        if self.cls == similarities.Similarity:
            self.assertTrue(len(index.shards) == len(index2.shards))
            index.destroy()
        else:
            if isinstance(index, similarities.SparseMatrixSimilarity):
                index.index = index.index.todense()
                index2.index = index2.index.todense()
            self.assertTrue(numpy.allclose(index.index, index2.index))
            self.assertEqual(index.num_best, index2.num_best)

    def test_large(self):
        if self.cls == similarities.WmdSimilarity and (not POT_EXT):
            self.skipTest('POT not installed')
        fname = get_tmpfile('gensim_similarities.tst.pkl')
        index = self.factoryMethod()
        index.save(fname, sep_limit=0)
        index2 = self.cls.load(fname)
        if self.cls == similarities.Similarity:
            self.assertTrue(len(index.shards) == len(index2.shards))
            index.destroy()
        else:
            if isinstance(index, similarities.SparseMatrixSimilarity):
                index.index = index.index.todense()
                index2.index = index2.index.todense()
            self.assertTrue(numpy.allclose(index.index, index2.index))
            self.assertEqual(index.num_best, index2.num_best)

    def test_large_compressed(self):
        if self.cls == similarities.WmdSimilarity and (not POT_EXT):
            self.skipTest('POT not installed')
        fname = get_tmpfile('gensim_similarities.tst.pkl.gz')
        index = self.factoryMethod()
        index.save(fname, sep_limit=0)
        index2 = self.cls.load(fname, mmap=None)
        if self.cls == similarities.Similarity:
            self.assertTrue(len(index.shards) == len(index2.shards))
            index.destroy()
        else:
            if isinstance(index, similarities.SparseMatrixSimilarity):
                index.index = index.index.todense()
                index2.index = index2.index.todense()
            self.assertTrue(numpy.allclose(index.index, index2.index))
            self.assertEqual(index.num_best, index2.num_best)

    def test_mmap(self):
        if self.cls == similarities.WmdSimilarity and (not POT_EXT):
            self.skipTest('POT not installed')
        fname = get_tmpfile('gensim_similarities.tst.pkl')
        index = self.factoryMethod()
        index.save(fname, sep_limit=0)
        index2 = self.cls.load(fname, mmap='r')
        if self.cls == similarities.Similarity:
            self.assertTrue(len(index.shards) == len(index2.shards))
            index.destroy()
        else:
            if isinstance(index, similarities.SparseMatrixSimilarity):
                index.index = index.index.todense()
                index2.index = index2.index.todense()
            self.assertTrue(numpy.allclose(index.index, index2.index))
            self.assertEqual(index.num_best, index2.num_best)

    def test_mmap_compressed(self):
        if self.cls == similarities.WmdSimilarity and (not POT_EXT):
            self.skipTest('POT not installed')
        fname = get_tmpfile('gensim_similarities.tst.pkl.gz')
        index = self.factoryMethod()
        index.save(fname, sep_limit=0)
        self.assertRaises(IOError, self.cls.load, fname, mmap='r')