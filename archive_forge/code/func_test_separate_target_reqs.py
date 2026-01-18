import mistralclient.auth.keystone
from mistralclient.tests.unit.v2 import base
def test_separate_target_reqs(self):
    params = {'a': 1, 'target_b': 2, 'c': 3, 'target_d': 4, 'target_target': 5, 'param_target': 6}
    nontarget, target = self.keystone._separate_target_reqs(params)
    self.assertIn('a', nontarget)
    self.assertIn('c', nontarget)
    self.assertIn('param_target', nontarget)
    self.assertIn('b', target)
    self.assertIn('d', target)
    self.assertIn('target', target)