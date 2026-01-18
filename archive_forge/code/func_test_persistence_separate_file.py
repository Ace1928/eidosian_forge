import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_persistence_separate_file(self):
    """Tests whether the model is saved and loaded correctly when the arrays are stored separately."""
    model = PoincareModel(self.data, burn_in=0, negative=3)
    model.train(epochs=1)
    model.save(testfile(), sep_limit=1)
    loaded = PoincareModel.load(testfile())
    self.models_equal(model, loaded)