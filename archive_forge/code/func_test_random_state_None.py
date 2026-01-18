import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_random_state_None(self):
    np.random.seed(42)
    rv = np.random.random_sample()
    np.random.seed(42)
    assert rv == self.instantiate_np_random_state(None)
    random.seed(42)
    rv = random.random()
    random.seed(42)
    assert rv == self.instantiate_py_random_state(None)