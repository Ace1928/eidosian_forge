from unittest import mock
import ddt
from oslo_db import exception as db_exc
from oslotest import base
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import resources
def test_subscribe_multiple_callbacks(self):
    self.manager.subscribe(callback_1, resources.PORT, events.BEFORE_CREATE)
    self.manager.subscribe(callback_2, resources.PORT, events.BEFORE_CREATE)
    self.assertEqual(2, len(self.manager._index))
    self.assertEqual(1, len(self.manager._callbacks[resources.PORT][events.BEFORE_CREATE]))
    self.assertEqual(2, len(self.manager._callbacks[resources.PORT][events.BEFORE_CREATE][0][1]))