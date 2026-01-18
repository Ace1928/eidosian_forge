from unittest import mock
import ddt
from oslo_db import exception as db_exc
from oslotest import base
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import resources
def test_unsubscribe_during_iteration(self):

    def unsub(r, e, *a, **k):
        return self.manager.unsubscribe(unsub, r, e)
    self.manager.subscribe(unsub, resources.PORT, events.BEFORE_CREATE)
    self.manager.publish(resources.PORT, events.BEFORE_CREATE, mock.ANY, payload=self.event_payload)
    self.assertNotIn(unsub, self.manager._index)