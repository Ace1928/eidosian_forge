import pickle
import unittest
from traits.api import Dict, HasTraits, Int, List
class PickleValidatedDictTestCase(unittest.TestCase):

    def test_pickle_validated_dict(self):
        x = pickle.dumps(C())
        try:
            pickle.loads(x)
        except AttributeError as e:
            self.fail('Unpickling raised an AttributeError: %s' % e)