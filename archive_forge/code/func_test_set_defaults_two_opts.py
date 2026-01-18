import copy
from oslo_config import cfg
from oslotest import base as test_base
from oslo_policy import opts
def test_set_defaults_two_opts(self):
    opts._register(self.conf)
    self.assertEqual(False, self.conf.oslo_policy.enforce_scope)
    self.assertEqual(False, self.conf.oslo_policy.enforce_new_defaults)
    opts.set_defaults(self.conf, enforce_scope=True, enforce_new_defaults=True)
    self.assertEqual(True, self.conf.oslo_policy.enforce_scope)
    self.assertEqual(True, self.conf.oslo_policy.enforce_new_defaults)