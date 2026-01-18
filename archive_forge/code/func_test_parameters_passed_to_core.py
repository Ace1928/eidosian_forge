import os
from breezy import tests
from breezy.tests import features
from breezy.transport import memory
def test_parameters_passed_to_core(self):
    params = self.get_params_passed_to_core('selftest --list-only')
    self.assertTrue('list_only' in params[1])
    params = self.get_params_passed_to_core('selftest --list-only selftest')
    self.assertTrue('list_only' in params[1])
    params = self.get_params_passed_to_core(['selftest', '--list-only', '--exclude', 'selftest'])
    self.assertTrue('list_only' in params[1])
    params = self.get_params_passed_to_core(['selftest', '--list-only', 'selftest', '--randomize', 'now'])
    self.assertSubset(['list_only', 'random_seed'], params[1])