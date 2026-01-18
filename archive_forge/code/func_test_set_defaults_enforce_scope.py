import copy
from oslo_config import cfg
from oslotest import base as test_base
from oslo_policy import opts
def test_set_defaults_enforce_scope(self):
    opts._register(self.conf)
    self.assertEqual(False, self.conf.oslo_policy.enforce_scope)
    opts.set_defaults(self.conf, enforce_scope=True)
    self.assertEqual(True, self.conf.oslo_policy.enforce_scope)