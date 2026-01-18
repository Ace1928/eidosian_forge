import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_training_multiple(self):
    """Tests that calling train multiple times results in different vectors."""
    model = PoincareModel(self.data_large, burn_in=0, negative=3)
    model.train(epochs=2)
    old_vectors = np.copy(model.kv.vectors)
    model.train(epochs=1)
    self.assertFalse(np.allclose(old_vectors, model.kv.vectors))
    old_vectors = np.copy(model.kv.vectors)
    model.train(epochs=0)
    self.assertTrue(np.allclose(old_vectors, model.kv.vectors))