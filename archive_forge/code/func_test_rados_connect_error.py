import io
from unittest import mock
from oslo_config import cfg
from oslo_utils import units
import glance_store as store
from glance_store._drivers import rbd as rbd_store
from glance_store import exceptions
from glance_store import location as g_location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
@mock.patch.object(MockRados.Rados, 'connect', side_effect=MockRados.Error)
def test_rados_connect_error(self, _):
    rbd_store.rados.Error = MockRados.Error
    rbd_store.rados.ObjectNotFound = MockRados.ObjectNotFound

    def test():
        with self.store.get_connection('conffile', 'rados_id'):
            pass
    self.assertRaises(exceptions.BadStoreConfiguration, test)