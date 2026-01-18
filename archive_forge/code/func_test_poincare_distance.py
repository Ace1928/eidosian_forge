import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_poincare_distance(self):
    """Test that poincare_distance returns correct distance between two input vectors."""
    vector_1 = self.vectors['dog.n.01']
    vector_2 = self.vectors['mammal.n.01']
    distance = self.vectors.vector_distance(vector_1, vector_2)
    self.assertTrue(np.allclose(distance, 4.5278745))
    distance = self.vectors.vector_distance(vector_1, vector_1)
    self.assertTrue(np.allclose(distance, 0))