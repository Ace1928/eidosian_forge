from unittest import mock
from openstack.baremetal.v1 import _proxy
from openstack.baremetal.v1 import allocation
from openstack.baremetal.v1 import chassis
from openstack.baremetal.v1 import driver
from openstack.baremetal.v1 import node
from openstack.baremetal.v1 import port
from openstack.baremetal.v1 import port_group
from openstack.baremetal.v1 import volume_connector
from openstack.baremetal.v1 import volume_target
from openstack import exceptions
from openstack.tests.unit import base
from openstack.tests.unit import test_proxy_base
class TestVolumeConnector(TestBaremetalProxy):

    def test_create_volume_connector(self):
        self.verify_create(self.proxy.create_volume_connector, volume_connector.VolumeConnector)

    def test_find_volume_connector(self):
        self.verify_find(self.proxy.find_volume_connector, volume_connector.VolumeConnector)

    def test_get_volume_connector(self):
        self.verify_get(self.proxy.get_volume_connector, volume_connector.VolumeConnector, mock_method=_MOCK_METHOD, expected_kwargs={'fields': None})

    def test_delete_volume_connector(self):
        self.verify_delete(self.proxy.delete_volume_connector, volume_connector.VolumeConnector, False)

    def test_delete_volume_connector_ignore(self):
        self.verify_delete(self.proxy.delete_volume_connector, volume_connector.VolumeConnector, True)