from unittest import mock
import ddt
from oslo_db import exception as db_exc
from oslotest import base
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import resources
def test_publish_invalid_payload(self):
    self.assertRaises(exceptions.Invalid, self.manager.publish, resources.PORT, events.AFTER_DELETE, self, payload=object())