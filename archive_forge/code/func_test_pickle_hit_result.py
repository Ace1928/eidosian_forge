import unittest
import pickle
from .common import MTurkCommon
def test_pickle_hit_result(self):
    result = self.create_hit_result()
    new_result = pickle.loads(pickle.dumps(result))