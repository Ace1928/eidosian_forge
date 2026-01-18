import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_poincare_distances_batch(self):
    """Test that poincare_distance_batch returns correct distances."""
    vector_1 = self.vectors['dog.n.01']
    vectors_2 = self.vectors[['mammal.n.01', 'dog.n.01']]
    distances = self.vectors.vector_distance_batch(vector_1, vectors_2)
    self.assertTrue(np.allclose(distances, [4.5278745, 0]))