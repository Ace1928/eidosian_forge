from tests.compat import unittest
from boto.ec2.connection import EC2Connection
from boto.ec2.blockdevicemapping import BlockDeviceType, BlockDeviceMapping
from tests.compat import OrderedDict
from tests.unit import AWSMockServiceTestCase
def test_startElement_with_name_virtualName_sets_and_returns_current_value(self):
    retval = self.block_device_mapping.startElement('virtualName', None, None)
    assert self.block_device_type_eq(retval, BlockDeviceType(self.block_device_mapping))