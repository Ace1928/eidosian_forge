import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_persistence_old_model(self):
    """Tests whether model from older gensim version is loaded correctly."""
    loaded = PoincareModel.load(datapath('poincare_test_3.4.0'))
    self.assertEqual(loaded.kv.vectors.shape, (239, 2))
    self.assertEqual(len(loaded.kv), 239)
    self.assertEqual(loaded.size, 2)
    self.assertEqual(len(loaded.all_relations), 200)