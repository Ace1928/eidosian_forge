import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
class TestPoincareKeyedVectors(unittest.TestCase):

    def setUp(self):
        self.vectors = PoincareKeyedVectors.load_word2vec_format(datapath('poincare_vectors.bin'), binary=True)

    def test_most_similar(self):
        """Test most_similar returns expected results."""
        expected = ['canine.n.02', 'hunting_dog.n.01', 'carnivore.n.01', 'placental.n.01', 'mammal.n.01']
        predicted = [result[0] for result in self.vectors.most_similar('dog.n.01', topn=5)]
        self.assertEqual(expected, predicted)

    def test_most_similar_topn(self):
        """Test most_similar returns correct results when `topn` is specified."""
        self.assertEqual(len(self.vectors.most_similar('dog.n.01', topn=5)), 5)
        self.assertEqual(len(self.vectors.most_similar('dog.n.01', topn=10)), 10)
        predicted = self.vectors.most_similar('dog.n.01', topn=None)
        self.assertEqual(len(predicted), len(self.vectors) - 1)
        self.assertEqual(predicted[-1][0], 'gallant_fox.n.01')

    def test_most_similar_raises_keyerror(self):
        """Test most_similar raises KeyError when input is out of vocab."""
        with self.assertRaises(KeyError):
            self.vectors.most_similar('not_in_vocab')

    def test_most_similar_restrict_vocab(self):
        """Test most_similar returns handles restrict_vocab correctly."""
        expected = set(self.vectors.index_to_key[:5])
        predicted = set((result[0] for result in self.vectors.most_similar('dog.n.01', topn=5, restrict_vocab=5)))
        self.assertEqual(expected, predicted)

    def test_most_similar_to_given(self):
        """Test most_similar_to_given returns correct results."""
        predicted = self.vectors.most_similar_to_given('dog.n.01', ['carnivore.n.01', 'placental.n.01', 'mammal.n.01'])
        self.assertEqual(predicted, 'carnivore.n.01')

    def test_most_similar_with_vector_input(self):
        """Test most_similar returns expected results with an input vector instead of an input word."""
        expected = ['dog.n.01', 'canine.n.02', 'hunting_dog.n.01', 'carnivore.n.01', 'placental.n.01']
        input_vector = self.vectors['dog.n.01']
        predicted = [result[0] for result in self.vectors.most_similar([input_vector], topn=5)]
        self.assertEqual(expected, predicted)

    def test_distance(self):
        """Test that distance returns expected values."""
        self.assertTrue(np.allclose(self.vectors.distance('dog.n.01', 'mammal.n.01'), 4.5278745))
        self.assertEqual(self.vectors.distance('dog.n.01', 'dog.n.01'), 0)

    def test_distances(self):
        """Test that distances between one word and multiple other words have expected values."""
        distances = self.vectors.distances('dog.n.01', ['mammal.n.01', 'dog.n.01'])
        self.assertTrue(np.allclose(distances, [4.5278745, 0]))
        distances = self.vectors.distances('dog.n.01')
        self.assertEqual(len(distances), len(self.vectors))
        self.assertTrue(np.allclose(distances[-1], 10.04756))

    def test_distances_with_vector_input(self):
        """Test that distances between input vector and a list of words have expected values."""
        input_vector = self.vectors['dog.n.01']
        distances = self.vectors.distances(input_vector, ['mammal.n.01', 'dog.n.01'])
        self.assertTrue(np.allclose(distances, [4.5278745, 0]))
        distances = self.vectors.distances(input_vector)
        self.assertEqual(len(distances), len(self.vectors))
        self.assertTrue(np.allclose(distances[-1], 10.04756))

    def test_poincare_distances_batch(self):
        """Test that poincare_distance_batch returns correct distances."""
        vector_1 = self.vectors['dog.n.01']
        vectors_2 = self.vectors[['mammal.n.01', 'dog.n.01']]
        distances = self.vectors.vector_distance_batch(vector_1, vectors_2)
        self.assertTrue(np.allclose(distances, [4.5278745, 0]))

    def test_poincare_distance(self):
        """Test that poincare_distance returns correct distance between two input vectors."""
        vector_1 = self.vectors['dog.n.01']
        vector_2 = self.vectors['mammal.n.01']
        distance = self.vectors.vector_distance(vector_1, vector_2)
        self.assertTrue(np.allclose(distance, 4.5278745))
        distance = self.vectors.vector_distance(vector_1, vector_1)
        self.assertTrue(np.allclose(distance, 0))

    def test_closest_child(self):
        """Test closest_child returns expected value and returns None for lowest node in hierarchy."""
        self.assertEqual(self.vectors.closest_child('dog.n.01'), 'terrier.n.01')
        self.assertEqual(self.vectors.closest_child('harbor_porpoise.n.01'), None)

    def test_closest_parent(self):
        """Test closest_parent returns expected value and returns None for highest node in hierarchy."""
        self.assertEqual(self.vectors.closest_parent('dog.n.01'), 'canine.n.02')
        self.assertEqual(self.vectors.closest_parent('mammal.n.01'), None)

    def test_ancestors(self):
        """Test ancestors returns expected list and returns empty list for highest node in hierarchy."""
        expected = ['canine.n.02', 'carnivore.n.01', 'placental.n.01', 'mammal.n.01']
        self.assertEqual(self.vectors.ancestors('dog.n.01'), expected)
        expected = []
        self.assertEqual(self.vectors.ancestors('mammal.n.01'), expected)

    def test_descendants(self):
        """Test descendants returns expected list and returns empty list for lowest node in hierarchy."""
        expected = ['terrier.n.01', 'sporting_dog.n.01', 'spaniel.n.01', 'water_spaniel.n.01', 'irish_water_spaniel.n.01']
        self.assertEqual(self.vectors.descendants('dog.n.01'), expected)
        self.assertEqual(self.vectors.descendants('dog.n.01', max_depth=3), expected[:3])

    def test_similarity(self):
        """Test similarity returns expected value for two nodes, and for identical nodes."""
        self.assertTrue(np.allclose(self.vectors.similarity('dog.n.01', 'dog.n.01'), 1))
        self.assertTrue(np.allclose(self.vectors.similarity('dog.n.01', 'mammal.n.01'), 0.180901358))

    def norm(self):
        """Test norm returns expected value."""
        self.assertTrue(np.allclose(self.vectors.norm('dog.n.01'), 0.97757602))
        self.assertTrue(np.allclose(self.vectors.norm('mammal.n.01'), 0.03914723))

    def test_difference_in_hierarchy(self):
        """Test difference_in_hierarchy returns expected value for two nodes, and for identical nodes."""
        self.assertTrue(np.allclose(self.vectors.difference_in_hierarchy('dog.n.01', 'dog.n.01'), 0))
        self.assertTrue(np.allclose(self.vectors.difference_in_hierarchy('mammal.n.01', 'dog.n.01'), 0.9384287))
        self.assertTrue(np.allclose(self.vectors.difference_in_hierarchy('dog.n.01', 'mammal.n.01'), -0.9384287))

    def test_closer_than(self):
        """Test closer_than returns expected value for distinct and identical nodes."""
        self.assertEqual(self.vectors.closer_than('dog.n.01', 'dog.n.01'), [])
        expected = set(['canine.n.02', 'hunting_dog.n.01'])
        self.assertEqual(set(self.vectors.closer_than('dog.n.01', 'carnivore.n.01')), expected)

    def test_rank(self):
        """Test rank returns expected value for distinct and identical nodes."""
        self.assertEqual(self.vectors.rank('dog.n.01', 'dog.n.01'), 1)
        self.assertEqual(self.vectors.rank('dog.n.01', 'carnivore.n.01'), 3)