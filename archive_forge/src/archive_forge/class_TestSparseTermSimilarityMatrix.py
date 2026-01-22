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
class TestSparseTermSimilarityMatrix(unittest.TestCase):

    def setUp(self):
        self.documents = [[u'government', u'denied', u'holiday'], [u'government', u'denied', u'holiday', u'slowing', u'hollingworth']]
        self.dictionary = Dictionary(self.documents)
        self.tfidf = TfidfModel(dictionary=self.dictionary)
        zero_index = UniformTermSimilarityIndex(self.dictionary, term_similarity=0.0)
        self.index = UniformTermSimilarityIndex(self.dictionary, term_similarity=0.5)
        self.identity_matrix = SparseTermSimilarityMatrix(zero_index, self.dictionary)
        self.uniform_matrix = SparseTermSimilarityMatrix(self.index, self.dictionary)
        self.vec1 = self.dictionary.doc2bow([u'government', u'government', u'denied'])
        self.vec2 = self.dictionary.doc2bow([u'government', u'holiday'])

    def test_empty_dictionary(self):
        with self.assertRaises(ValueError):
            SparseTermSimilarityMatrix(self.index, [])

    def test_type(self):
        """Test the type of the produced matrix."""
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary).matrix
        self.assertTrue(isinstance(matrix, scipy.sparse.csc_matrix))

    def test_diagonal(self):
        """Test the existence of ones on the main diagonal."""
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary).matrix.todense()
        self.assertTrue(numpy.all(numpy.diag(matrix) == numpy.ones(matrix.shape[0])))

    def test_order(self):
        """Test the matrix order."""
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary).matrix.todense()
        self.assertEqual(matrix.shape[0], len(self.dictionary))
        self.assertEqual(matrix.shape[1], len(self.dictionary))

    def test_dtype(self):
        """Test the dtype parameter of the matrix constructor."""
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, dtype=numpy.float32).matrix.todense()
        self.assertEqual(numpy.float32, matrix.dtype)
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, dtype=numpy.float64).matrix.todense()
        self.assertEqual(numpy.float64, matrix.dtype)

    def test_nonzero_limit(self):
        """Test the nonzero_limit parameter of the matrix constructor."""
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, nonzero_limit=100).matrix.todense()
        self.assertGreaterEqual(101, numpy.max(numpy.sum(matrix != 0, axis=0)))
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, nonzero_limit=4).matrix.todense()
        self.assertGreaterEqual(5, numpy.max(numpy.sum(matrix != 0, axis=0)))
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, nonzero_limit=1).matrix.todense()
        self.assertGreaterEqual(2, numpy.max(numpy.sum(matrix != 0, axis=0)))
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, nonzero_limit=0).matrix.todense()
        self.assertEqual(1, numpy.max(numpy.sum(matrix != 0, axis=0)))
        self.assertTrue(numpy.all(matrix == numpy.eye(matrix.shape[0])))

    def test_symmetric(self):
        """Test the symmetric parameter of the matrix constructor."""
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary).matrix.todense()
        self.assertTrue(numpy.all(matrix == matrix.T))
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, nonzero_limit=1).matrix.todense()
        expected_matrix = numpy.array([[1.0, 0.5, 0.0, 0.0, 0.0], [0.5, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
        self.assertTrue(numpy.all(expected_matrix == matrix))
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, nonzero_limit=1, symmetric=False).matrix.todense()
        expected_matrix = numpy.array([[1.0, 0.5, 0.5, 0.5, 0.5], [0.5, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
        self.assertTrue(numpy.all(expected_matrix == matrix))

    def test_dominant(self):
        """Test the dominant parameter of the matrix constructor."""
        negative_index = UniformTermSimilarityIndex(self.dictionary, term_similarity=-0.5)
        matrix = SparseTermSimilarityMatrix(negative_index, self.dictionary, nonzero_limit=2).matrix.todense()
        expected_matrix = numpy.array([[1.0, -0.5, -0.5, 0.0, 0.0], [-0.5, 1.0, 0.0, -0.5, 0.0], [-0.5, 0.0, 1.0, 0.0, 0.0], [0.0, -0.5, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
        self.assertTrue(numpy.all(expected_matrix == matrix))
        matrix = SparseTermSimilarityMatrix(negative_index, self.dictionary, nonzero_limit=2, dominant=True).matrix.todense()
        expected_matrix = numpy.array([[1.0, -0.5, 0.0, 0.0, 0.0], [-0.5, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
        self.assertTrue(numpy.all(expected_matrix == matrix))

    def test_tfidf(self):
        """Test the tfidf parameter of the matrix constructor."""
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, nonzero_limit=1).matrix.todense()
        expected_matrix = numpy.array([[1.0, 0.5, 0.0, 0.0, 0.0], [0.5, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
        self.assertTrue(numpy.all(expected_matrix == matrix))
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, nonzero_limit=1, tfidf=self.tfidf).matrix.todense()
        expected_matrix = numpy.array([[1.0, 0.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.5, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
        self.assertTrue(numpy.all(expected_matrix == matrix))

    def test_encapsulation(self):
        """Test the matrix encapsulation."""
        expected_matrix = numpy.array([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]])
        matrix = SparseTermSimilarityMatrix(scipy.sparse.csc_matrix(expected_matrix)).matrix
        self.assertTrue(isinstance(matrix, scipy.sparse.csc_matrix))
        self.assertTrue(numpy.all(matrix.todense() == expected_matrix))
        matrix = SparseTermSimilarityMatrix(scipy.sparse.csr_matrix(expected_matrix)).matrix
        self.assertTrue(isinstance(matrix, scipy.sparse.csc_matrix))
        self.assertTrue(numpy.all(matrix.todense() == expected_matrix))

    def test_inner_product_zerovector_zerovector_default(self):
        """Test the inner product between two zero vectors with the default normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product([], []))

    def test_inner_product_zerovector_zerovector_false_maintain(self):
        """Test the inner product between two zero vectors with the (False, 'maintain') normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product([], [], normalized=(False, 'maintain')))

    def test_inner_product_zerovector_zerovector_false_true(self):
        """Test the inner product between two zero vectors with the (False, True) normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product([], [], normalized=(False, True)))

    def test_inner_product_zerovector_zerovector_maintain_false(self):
        """Test the inner product between two zero vectors with the ('maintain', False) normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product([], [], normalized=('maintain', False)))

    def test_inner_product_zerovector_zerovector_maintain_maintain(self):
        """Test the inner product between two zero vectors with the ('maintain', 'maintain') normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product([], [], normalized=('maintain', 'maintain')))

    def test_inner_product_zerovector_zerovector_maintain_true(self):
        """Test the inner product between two zero vectors with the ('maintain', True) normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product([], [], normalized=('maintain', True)))

    def test_inner_product_zerovector_zerovector_true_false(self):
        """Test the inner product between two zero vectors with the (True, False) normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product([], [], normalized=(True, False)))

    def test_inner_product_zerovector_zerovector_true_maintain(self):
        """Test the inner product between two zero vectors with the (True, 'maintain') normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product([], [], normalized=(True, 'maintain')))

    def test_inner_product_zerovector_zerovector_true_true(self):
        """Test the inner product between two zero vectors with the (True, True) normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product([], [], normalized=(True, True)))

    def test_inner_product_zerovector_vector_default(self):
        """Test the inner product between a zero vector and a vector with the default normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product([], self.vec2))

    def test_inner_product_zerovector_vector_false_maintain(self):
        """Test the inner product between a zero vector and a vector with the (False, 'maintain') normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product([], self.vec2, normalized=(False, 'maintain')))

    def test_inner_product_zerovector_vector_false_true(self):
        """Test the inner product between a zero vector and a vector with the (False, True) normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product([], self.vec2, normalized=(False, True)))

    def test_inner_product_zerovector_vector_maintain_false(self):
        """Test the inner product between a zero vector and a vector with the ('maintain', False) normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product([], self.vec2, normalized=('maintain', False)))

    def test_inner_product_zerovector_vector_maintain_maintain(self):
        """Test the inner product between a zero vector and a vector with the ('maintain', 'maintain') normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product([], self.vec2, normalized=('maintain', 'maintain')))

    def test_inner_product_zerovector_vector_maintain_true(self):
        """Test the inner product between a zero vector and a vector with the ('maintain', True) normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product([], self.vec2, normalized=('maintain', True)))

    def test_inner_product_zerovector_vector_true_false(self):
        """Test the inner product between a zero vector and a vector with the (True, False) normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product([], self.vec2, normalized=(True, False)))

    def test_inner_product_zerovector_vector_true_maintain(self):
        """Test the inner product between a zero vector and a vector with the (True, 'maintain') normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product([], self.vec2, normalized=(True, 'maintain')))

    def test_inner_product_zerovector_vector_true_true(self):
        """Test the inner product between a zero vector and a vector with the (True, True) normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product([], self.vec2, normalized=(True, True)))

    def test_inner_product_vector_zerovector_default(self):
        """Test the inner product between a vector and a zero vector with the default normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product(self.vec1, []))

    def test_inner_product_vector_zerovector_false_maintain(self):
        """Test the inner product between a vector and a zero vector with the (False, 'maintain') normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product(self.vec1, [], normalized=(False, 'maintain')))

    def test_inner_product_vector_zerovector_false_true(self):
        """Test the inner product between a vector and a zero vector with the (False, True) normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product(self.vec1, [], normalized=(False, True)))

    def test_inner_product_vector_zerovector_maintain_false(self):
        """Test the inner product between a vector and a zero vector with the ('maintain', False) normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product(self.vec1, [], normalized=('maintain', False)))

    def test_inner_product_vector_zerovector_maintain_maintain(self):
        """Test the inner product between a vector and a zero vector with the ('maintain', 'maintain') normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product(self.vec1, [], normalized=('maintain', 'maintain')))

    def test_inner_product_vector_zerovector_maintain_true(self):
        """Test the inner product between a vector and a zero vector with the ('maintain', True) normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product(self.vec1, [], normalized=('maintain', True)))

    def test_inner_product_vector_zerovector_true_false(self):
        """Test the inner product between a vector and a zero vector with the (True, False) normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product(self.vec1, [], normalized=(True, False)))

    def test_inner_product_vector_zerovector_true_maintain(self):
        """Test the inner product between a vector and a zero vector with the (True, 'maintain') normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product(self.vec1, [], normalized=(True, 'maintain')))

    def test_inner_product_vector_zerovector_true_true(self):
        """Test the inner product between a vector and a zero vector with the (True, True) normalization."""
        self.assertEqual(0.0, self.uniform_matrix.inner_product(self.vec1, [], normalized=(True, True)))

    def test_inner_product_vector_vector_default(self):
        """Test the inner product between two vectors with the default normalization."""
        expected_result = 0.0
        expected_result += 2 * 1.0 * 1
        expected_result += 2 * 0.5 * 1
        expected_result += 1 * 0.5 * 1
        expected_result += 1 * 0.5 * 1
        result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        self.assertAlmostEqual(expected_result, result, places=5)

    def test_inner_product_vector_vector_false_maintain(self):
        """Test the inner product between two vectors with the (False, 'maintain') normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        result = self.uniform_matrix.inner_product(self.vec1, self.vec2, normalized=(False, 'maintain'))
        self.assertAlmostEqual(expected_result, result, places=5)

    def test_inner_product_vector_vector_false_true(self):
        """Test the inner product between two vectors with the (False, True) normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        result = self.uniform_matrix.inner_product(self.vec1, self.vec2, normalized=(False, True))
        self.assertAlmostEqual(expected_result, result, places=5)

    def test_inner_product_vector_vector_maintain_false(self):
        """Test the inner product between two vectors with the ('maintain', False) normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        result = self.uniform_matrix.inner_product(self.vec1, self.vec2, normalized=('maintain', False))
        self.assertAlmostEqual(expected_result, result, places=5)

    def test_inner_product_vector_vector_maintain_maintain(self):
        """Test the inner product between two vectors with the ('maintain', 'maintain') normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        result = self.uniform_matrix.inner_product(self.vec1, self.vec2, normalized=('maintain', 'maintain'))
        self.assertAlmostEqual(expected_result, result, places=5)

    def test_inner_product_vector_vector_maintain_true(self):
        """Test the inner product between two vectors with the ('maintain', True) normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        result = self.uniform_matrix.inner_product(self.vec1, self.vec2, normalized=('maintain', True))
        self.assertAlmostEqual(expected_result, result, places=5)

    def test_inner_product_vector_vector_true_false(self):
        """Test the inner product between two vectors with the (True, False) normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        result = self.uniform_matrix.inner_product(self.vec1, self.vec2, normalized=(True, False))
        self.assertAlmostEqual(expected_result, result, places=5)

    def test_inner_product_vector_vector_true_maintain(self):
        """Test the inner product between two vectors with the (True, 'maintain') normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        result = self.uniform_matrix.inner_product(self.vec1, self.vec2, normalized=(True, 'maintain'))
        self.assertAlmostEqual(expected_result, result, places=5)

    def test_inner_product_vector_vector_true_true(self):
        """Test the inner product between two vectors with the (True, True) normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        result = self.uniform_matrix.inner_product(self.vec1, self.vec2, normalized=(True, True))
        self.assertAlmostEqual(expected_result, result, places=5)

    def test_inner_product_vector_corpus_default(self):
        """Test the inner product between a vector and a corpus with the default normalization."""
        expected_result = 0.0
        expected_result += 2 * 1.0 * 1
        expected_result += 2 * 0.5 * 1
        expected_result += 1 * 0.5 * 1
        expected_result += 1 * 0.5 * 1
        expected_result = numpy.full((1, 2), expected_result)
        result = self.uniform_matrix.inner_product(self.vec1, [self.vec2] * 2)
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_vector_corpus_false_maintain(self):
        """Test the inner product between a vector and a corpus with the (False, 'maintain') normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((1, 2), expected_result)
        result = self.uniform_matrix.inner_product(self.vec1, [self.vec2] * 2, normalized=(False, 'maintain'))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_vector_corpus_false_true(self):
        """Test the inner product between a vector and a corpus with the (False, True) normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((1, 2), expected_result)
        result = self.uniform_matrix.inner_product(self.vec1, [self.vec2] * 2, normalized=(False, True))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_vector_corpus_maintain_false(self):
        """Test the inner product between a vector and a corpus with the ('maintain', False) normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result = numpy.full((1, 2), expected_result)
        result = self.uniform_matrix.inner_product(self.vec1, [self.vec2] * 2, normalized=('maintain', False))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_vector_corpus_maintain_maintain(self):
        """Test the inner product between a vector and a corpus with the ('maintain', 'maintain') normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((1, 2), expected_result)
        result = self.uniform_matrix.inner_product(self.vec1, [self.vec2] * 2, normalized=('maintain', 'maintain'))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_vector_corpus_maintain_true(self):
        """Test the inner product between a vector and a corpus with the ('maintain', True) normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((1, 2), expected_result)
        result = self.uniform_matrix.inner_product(self.vec1, [self.vec2] * 2, normalized=('maintain', True))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_vector_corpus_true_false(self):
        """Test the inner product between a vector and a corpus with the (True, False) normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result = numpy.full((1, 2), expected_result)
        result = self.uniform_matrix.inner_product(self.vec1, [self.vec2] * 2, normalized=(True, False))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_vector_corpus_true_maintain(self):
        """Test the inner product between a vector and a corpus with the (True, 'maintain') normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((1, 2), expected_result)
        result = self.uniform_matrix.inner_product(self.vec1, [self.vec2] * 2, normalized=(True, 'maintain'))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_vector_corpus_true_true(self):
        """Test the inner product between a vector and a corpus with the (True, True) normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((1, 2), expected_result)
        result = self.uniform_matrix.inner_product(self.vec1, [self.vec2] * 2, normalized=(True, True))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_corpus_vector_default(self):
        """Test the inner product between a corpus and a vector with the default normalization."""
        expected_result = 0.0
        expected_result += 2 * 1.0 * 1
        expected_result += 2 * 0.5 * 1
        expected_result += 1 * 0.5 * 1
        expected_result += 1 * 0.5 * 1
        expected_result = numpy.full((3, 1), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, self.vec2)
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_corpus_vector_false_maintain(self):
        """Test the inner product between a corpus and a vector with the (False, 'maintain') normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 1), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, self.vec2, normalized=(False, 'maintain'))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_corpus_vector_false_true(self):
        """Test the inner product between a corpus and a vector with the (False, True) normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 1), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, self.vec2, normalized=(False, True))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_corpus_vector_maintain_false(self):
        """Test the inner product between a corpus and a vector with the ('maintain', False) normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result = numpy.full((3, 1), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, self.vec2, normalized=('maintain', False))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_corpus_vector_maintain_maintain(self):
        """Test the inner product between a corpus and a vector with the ('maintain', 'maintain') normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 1), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, self.vec2, normalized=('maintain', 'maintain'))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_corpus_vector_maintain_true(self):
        """Test the inner product between a corpus and a vector with the ('maintain', True) normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 1), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, self.vec2, normalized=('maintain', True))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_corpus_vector_true_false(self):
        """Test the inner product between a corpus and a vector with the (True, False) normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result = numpy.full((3, 1), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, self.vec2, normalized=(True, False))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_corpus_vector_true_maintain(self):
        """Test the inner product between a corpus and a vector with the (True, 'maintain') normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 1), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, self.vec2, normalized=(True, 'maintain'))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_corpus_vector_true_true(self):
        """Test the inner product between a corpus and a vector with the (True, True) normalization."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 1), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, self.vec2, normalized=(True, True))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_corpus_corpus_default(self):
        """Test the inner product between two corpora with the default normalization."""
        expected_result = 0.0
        expected_result += 2 * 1.0 * 1
        expected_result += 2 * 0.5 * 1
        expected_result += 1 * 0.5 * 1
        expected_result += 1 * 0.5 * 1
        expected_result = numpy.full((3, 2), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, [self.vec2] * 2)
        self.assertTrue(isinstance(result, scipy.sparse.csr_matrix))
        self.assertTrue(numpy.allclose(expected_result, result.todense()))

    def test_inner_product_corpus_corpus_false_maintain(self):
        """Test the inner product between two corpora with the (False, 'maintain')."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 2), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, [self.vec2] * 2, normalized=(False, 'maintain'))
        self.assertTrue(isinstance(result, scipy.sparse.csr_matrix))
        self.assertTrue(numpy.allclose(expected_result, result.todense()))

    def test_inner_product_corpus_corpus_false_true(self):
        """Test the inner product between two corpora with the (False, True)."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 2), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, [self.vec2] * 2, normalized=(False, True))
        self.assertTrue(isinstance(result, scipy.sparse.csr_matrix))
        self.assertTrue(numpy.allclose(expected_result, result.todense()))

    def test_inner_product_corpus_corpus_maintain_false(self):
        """Test the inner product between two corpora with the ('maintain', False)."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result = numpy.full((3, 2), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, [self.vec2] * 2, normalized=('maintain', False))
        self.assertTrue(isinstance(result, scipy.sparse.csr_matrix))
        self.assertTrue(numpy.allclose(expected_result, result.todense()))

    def test_inner_product_corpus_corpus_maintain_maintain(self):
        """Test the inner product between two corpora with the ('maintain', 'maintain')."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 2), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, [self.vec2] * 2, normalized=('maintain', 'maintain'))
        self.assertTrue(isinstance(result, scipy.sparse.csr_matrix))
        self.assertTrue(numpy.allclose(expected_result, result.todense()))

    def test_inner_product_corpus_corpus_maintain_true(self):
        """Test the inner product between two corpora with the ('maintain', True)."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 2), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, [self.vec2] * 2, normalized=('maintain', True))
        self.assertTrue(isinstance(result, scipy.sparse.csr_matrix))
        self.assertTrue(numpy.allclose(expected_result, result.todense()))

    def test_inner_product_corpus_corpus_true_false(self):
        """Test the inner product between two corpora with the (True, False)."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result = numpy.full((3, 2), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, [self.vec2] * 2, normalized=(True, False))
        self.assertTrue(isinstance(result, scipy.sparse.csr_matrix))
        self.assertTrue(numpy.allclose(expected_result, result.todense()))

    def test_inner_product_corpus_corpus_true_maintain(self):
        """Test the inner product between two corpora with the (True, 'maintain')."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 2), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, [self.vec2] * 2, normalized=(True, 'maintain'))
        self.assertTrue(isinstance(result, scipy.sparse.csr_matrix))
        self.assertTrue(numpy.allclose(expected_result, result.todense()))

    def test_inner_product_corpus_corpus_true_true(self):
        """Test the inner product between two corpora with the (True, True)."""
        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 2), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, [self.vec2] * 2, normalized=(True, True))
        self.assertTrue(isinstance(result, scipy.sparse.csr_matrix))
        self.assertTrue(numpy.allclose(expected_result, result.todense()))