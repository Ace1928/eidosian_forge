import contextlib
import datetime
from unittest import mock
import uuid
import warnings
from openstack.block_storage.v3 import volume
from openstack.compute.v2 import _proxy
from openstack.compute.v2 import aggregate
from openstack.compute.v2 import availability_zone as az
from openstack.compute.v2 import extension
from openstack.compute.v2 import flavor
from openstack.compute.v2 import hypervisor
from openstack.compute.v2 import image
from openstack.compute.v2 import keypair
from openstack.compute.v2 import migration
from openstack.compute.v2 import quota_set
from openstack.compute.v2 import server
from openstack.compute.v2 import server_action
from openstack.compute.v2 import server_group
from openstack.compute.v2 import server_interface
from openstack.compute.v2 import server_ip
from openstack.compute.v2 import server_migration
from openstack.compute.v2 import server_remote_console
from openstack.compute.v2 import service
from openstack.compute.v2 import usage
from openstack.compute.v2 import volume_attachment
from openstack import resource
from openstack.tests.unit import test_proxy_base
from openstack import warnings as os_warnings
def test_server_rebuild(self):
    id = 'test_image_id'
    image_obj = image.Image(id='test_image_id')
    self._verify('openstack.compute.v2.server.Server.rebuild', self.proxy.rebuild_server, method_args=['value'], method_kwargs={'name': 'test_server', 'admin_password': 'test_pass', 'metadata': {'k1': 'v1'}, 'image': image_obj}, expected_args=[self.proxy], expected_kwargs={'name': 'test_server', 'admin_password': 'test_pass', 'metadata': {'k1': 'v1'}, 'image': image_obj})
    self._verify('openstack.compute.v2.server.Server.rebuild', self.proxy.rebuild_server, method_args=['value'], method_kwargs={'name': 'test_server', 'admin_password': 'test_pass', 'metadata': {'k1': 'v1'}, 'image': id}, expected_args=[self.proxy], expected_kwargs={'name': 'test_server', 'admin_password': 'test_pass', 'metadata': {'k1': 'v1'}, 'image': id})