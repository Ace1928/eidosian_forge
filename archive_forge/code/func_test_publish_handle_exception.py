from unittest import mock
import ddt
from oslo_db import exception as db_exc
from oslotest import base
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import resources
def test_publish_handle_exception(self):
    self.manager.subscribe(callback_raise, resources.PORT, events.BEFORE_CREATE)
    e = self.assertRaises(exceptions.CallbackFailure, self.manager.publish, resources.PORT, events.BEFORE_CREATE, self, payload=self.event_payload)
    self.assertIsInstance(e.errors[0], exceptions.NotificationError)