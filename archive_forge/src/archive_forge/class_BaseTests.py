import sys
import unittest
from libcloud.common.base import Connection, ConnectionKey, ConnectionUserAndKey
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import StorageVolumeState
class BaseTests(unittest.TestCase):

    def test_base_node(self):
        Node(id=0, name=0, state=0, public_ips=0, private_ips=0, driver=FakeDriver())

    def test_base_node_size(self):
        NodeSize(id=0, name=0, ram=0, disk=0, bandwidth=0, price=0, driver=FakeDriver())

    def test_base_node_image(self):
        NodeImage(id=0, name=0, driver=FakeDriver())

    def test_base_storage_volume(self):
        StorageVolume(id='0', name='0', size=10, driver=FakeDriver(), state=StorageVolumeState.AVAILABLE)

    def test_base_node_driver(self):
        NodeDriver('foo')

    def test_base_connection_key(self):
        ConnectionKey('foo')

    def test_base_connection_userkey(self):
        ConnectionUserAndKey('foo', 'bar')

    def test_base_connection_timeout(self):
        Connection(timeout=10)