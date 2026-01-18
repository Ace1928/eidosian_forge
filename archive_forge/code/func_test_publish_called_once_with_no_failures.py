from unittest import mock
import ddt
from oslo_db import exception as db_exc
from oslotest import base
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import resources
def test_publish_called_once_with_no_failures(self):
    with mock.patch.object(self.manager, '_notify_loop') as n:
        n.return_value = False
        self.manager.publish(resources.PORT, events.BEFORE_CREATE, mock.ANY, payload=self.event_payload)
        n.assert_called_once_with(resources.PORT, events.BEFORE_CREATE, mock.ANY, self.event_payload)