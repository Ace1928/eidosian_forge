from unittest import mock
from oslotest import base
import testtools
from neutron_lib.callbacks import events
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import registry
from neutron_lib.callbacks import resources
from neutron_lib import fixture
def test_object_inheriting_others_no_double_subscribe(self):
    with mock.patch.object(registry, 'subscribe') as sub:
        callback = AnotherObjectWithDecoratedCallback()
        priority_call = [mock.call(callback.callback2, resources.NETWORK, events.AFTER_DELETE, PRI_CALLBACK)]
        self.assertEqual(4, len(sub.mock_calls))
        sub.assert_has_calls(priority_call)