from unittest import mock
from neutron_lib.db import model_query
from neutron_lib import fixture
from neutron_lib.tests import _base
from neutron_lib.utils import helpers
def test_register_hook_non_callables(self):
    mock_model = mock.Mock()
    model_query.register_hook(mock_model, 'hook1', self._mock_hook, {}, result_filters={})
    self.assertEqual(1, len(model_query._model_query_hooks.keys()))
    hook_ref = helpers.make_weak_ref(self._mock_hook)
    registered_hooks = model_query.get_hooks(mock_model)
    self.assertEqual(1, len(registered_hooks))
    for d in registered_hooks:
        for k in d.keys():
            if k == 'query':
                self.assertEqual(hook_ref, d.get(k))
            else:
                self.assertEqual({}, d.get(k))