import stevedore
from testtools import matchers
from glance_store import backend
from glance_store.tests import base
def _test_entry_point(self, namespace, expected_opt_groups, expected_opt_names):
    opt_list = None
    mgr = stevedore.NamedExtensionManager('oslo.config.opts', names=[namespace], invoke_on_load=False, on_load_failure_callback=on_load_failure_callback)
    for ext in mgr:
        list_fn = ext.plugin
        opt_list = list_fn()
        break
    self.assertIsNotNone(opt_list)
    self._check_opt_groups(opt_list, expected_opt_groups)
    self._check_opt_names(opt_list, expected_opt_names)