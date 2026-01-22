from unittest import mock
import uuid
from cinderclient import exceptions as cinder_exc
from keystoneauth1 import exceptions as ks_exceptions
from heat.common import exception
from heat.engine.clients.os import cinder
from heat.tests import common
from heat.tests import utils
class CinderClientPluginTest(common.HeatTestCase):
    """Basic tests for :module:'heat.engine.clients.os.cinder'."""

    def setUp(self):
        super(CinderClientPluginTest, self).setUp()
        self.cinder_client = mock.MagicMock()
        con = utils.dummy_context()
        c = con.clients
        self.cinder_plugin = c.client_plugin('cinder')
        self.cinder_plugin.client = lambda: self.cinder_client

    def test_get_volume(self):
        """Tests the get_volume function."""
        volume_id = str(uuid.uuid4())
        my_volume = mock.MagicMock()
        self.cinder_client.volumes.get.return_value = my_volume
        self.assertEqual(my_volume, self.cinder_plugin.get_volume(volume_id))
        self.cinder_client.volumes.get.assert_called_once_with(volume_id)

    def test_get_snapshot(self):
        """Tests the get_volume_snapshot function."""
        snapshot_id = str(uuid.uuid4())
        my_snapshot = mock.MagicMock()
        self.cinder_client.volume_snapshots.get.return_value = my_snapshot
        self.assertEqual(my_snapshot, self.cinder_plugin.get_volume_snapshot(snapshot_id))
        self.cinder_client.volume_snapshots.get.assert_called_once_with(snapshot_id)