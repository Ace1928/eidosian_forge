import hashlib
import io
from unittest import mock
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers import rbd as rbd_store
from glance_store import exceptions
from glance_store import location as g_location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
from glance_store.tests import utils as test_utils
@mock.patch.object(rbd_store, 'rbd')
@mock.patch.object(rbd_store, 'tpool')
@mock.patch('oslo_utils.eventletutils.is_monkey_patched')
def test_rbd_proxy(self, mock_patched, mock_tpool, mock_rbd):
    mock_patched.return_value = False
    self.assertEqual(mock_rbd.RBD(), self.store.RBDProxy())
    mock_patched.return_value = True
    self.assertEqual(mock_tpool.Proxy.return_value, self.store.RBDProxy())