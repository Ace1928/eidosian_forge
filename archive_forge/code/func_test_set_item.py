import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_set_item(self):
    """Test that __setitem__ works correctly."""
    vocab_size = len(self.vectors)
    entity = '___some_new_entity___'
    vector = np.random.randn(self.vectors.vector_size)
    self.vectors[entity] = vector
    self.assertEqual(len(self.vectors), vocab_size + 1)
    self.assertTrue(np.allclose(self.vectors[entity], vector))
    vocab_size = len(self.vectors)
    vector = np.random.randn(self.vectors.vector_size)
    self.vectors['war'] = vector
    self.assertEqual(len(self.vectors), vocab_size)
    self.assertTrue(np.allclose(self.vectors['war'], vector))
    vocab_size = len(self.vectors)
    entities = ['war', '___some_new_entity1___', '___some_new_entity2___', 'terrorism', 'conflict']
    vectors = [np.random.randn(self.vectors.vector_size) for _ in range(len(entities))]
    self.vectors[entities] = vectors
    self.assertEqual(len(self.vectors), vocab_size + 2)
    for ent, vector in zip(entities, vectors):
        self.assertTrue(np.allclose(self.vectors[ent], vector))