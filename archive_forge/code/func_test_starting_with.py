import os
from breezy import tests
from breezy.tests import features
from breezy.transport import memory
def test_starting_with(self):
    params = self.get_params_passed_to_core('selftest --starting-with foo')
    self.assertEqual(['foo'], params[1]['starting_with'])