import pathlib
import pickle
import unittest
from traits.testing.optional_dependencies import (
@requires_pkg_resources
def test_unpickling_historical_pickles(self):
    for pickle_path in find_pickles():
        with self.subTest(filename=pickle_path.name):
            with pickle_path.open('rb') as f:
                pickle.load(f)