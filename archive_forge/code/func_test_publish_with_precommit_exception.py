from unittest import mock
import ddt
from oslo_db import exception as db_exc
from oslotest import base
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import resources
def test_publish_with_precommit_exception(self):
    with mock.patch.object(self.manager, '_notify_loop') as n:
        n.return_value = ['error']
        self.assertRaises(exceptions.CallbackFailure, self.manager.publish, mock.ANY, events.PRECOMMIT_UPDATE, mock.ANY, payload=self.event_payload)
        expected_calls = [mock.call(mock.ANY, 'precommit_update', mock.ANY, self.event_payload)]
        n.assert_has_calls(expected_calls)