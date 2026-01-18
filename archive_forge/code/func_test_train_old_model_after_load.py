import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_train_old_model_after_load(self):
    """Tests whether loaded model from older gensim version can be trained correctly."""
    loaded = PoincareModel.load(datapath('poincare_test_3.4.0'))
    old_vectors = np.copy(loaded.kv.vectors)
    loaded.train(epochs=2)
    self.assertFalse(np.allclose(old_vectors, loaded.kv.vectors))