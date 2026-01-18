import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_random_state_np_random_RandomState(self):
    np.random.seed(42)
    np_rv = np.random.random_sample()
    np.random.seed(42)
    seed = 1
    rng = np.random.RandomState(seed)
    rval = self.instantiate_np_random_state(seed)
    rval_expected = np.random.RandomState(seed).rand()
    assert rval, rval_expected
    rval = self.instantiate_py_random_state(seed)
    rval_expected = np.random.RandomState(seed).rand()
    assert rval, rval_expected
    assert np_rv == np.random.random_sample()