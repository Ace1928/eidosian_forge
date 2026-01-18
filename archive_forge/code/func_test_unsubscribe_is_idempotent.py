from unittest import mock
import ddt
from oslo_db import exception as db_exc
from oslotest import base
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import resources
@ddt.data(True, False)
def test_unsubscribe_is_idempotent(self, cancellable):
    self.manager.subscribe(callback_1, resources.PORT, events.BEFORE_CREATE, cancellable=cancellable)
    self.manager.unsubscribe(callback_1, resources.PORT, events.BEFORE_CREATE)
    self.manager.unsubscribe(callback_1, resources.PORT, events.BEFORE_CREATE)
    self.assertNotIn(callback_id_1, self.manager._index)
    self.assertNotIn(callback_id_1, self.manager._callbacks[resources.PORT][events.BEFORE_CREATE])