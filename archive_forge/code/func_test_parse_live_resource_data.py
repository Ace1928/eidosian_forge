from unittest import mock
from glanceclient import exc
from heat.common import exception
from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_parse_live_resource_data(self):
    resource_data = {'name': 'test', 'disk_format': 'qcow2', 'container_format': 'bare', 'active': None, 'protected': False, 'is_public': False, 'min_disk': 0, 'min_ram': 0, 'id': '41f0e60c-ebb4-4375-a2b4-845ae8b9c995', 'tags': [], 'architecture': 'test_architecture', 'kernel_id': '12345678-1234-1234-1234-123456789012', 'os_distro': 'new_distro', 'os_version': '1.0', 'os_secure_boot': 'False', 'owner': 'new_owner', 'hw_firmware_type': 'uefi', 'ramdisk_id': '12345678-1234-1234-1234-123456789012', 'members': None, 'visibility': 'private'}
    resource_properties = self.stack.t.t['resources']['my_image']['properties'].copy()
    resource_properties['extra_properties'] = {'hw_firmware_type': 'uefi', 'os_secure_boot': 'required'}
    reality = self.my_image.parse_live_resource_data(resource_properties, resource_data)
    expected = {'name': 'test', 'disk_format': 'qcow2', 'container_format': 'bare', 'active': None, 'protected': False, 'min_disk': 0, 'min_ram': 0, 'id': '41f0e60c-ebb4-4375-a2b4-845ae8b9c995', 'tags': [], 'architecture': 'test_architecture', 'kernel_id': '12345678-1234-1234-1234-123456789012', 'os_distro': 'new_distro', 'os_version': '1.0', 'owner': 'new_owner', 'ramdisk_id': '12345678-1234-1234-1234-123456789012', 'members': None, 'visibility': 'private', 'extra_properties': {'hw_firmware_type': 'uefi', 'os_secure_boot': 'False'}}
    self.assertEqual(set(expected.keys()), set(reality.keys()))
    for key in expected:
        self.assertEqual(expected[key], reality[key])
    for key in expected['extra_properties']:
        self.assertEqual(expected['extra_properties'][key], reality['extra_properties'][key])