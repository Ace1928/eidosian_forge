import io
from unittest import mock
from oslo_vmware import exceptions
from oslo_vmware import image_transfer
from oslo_vmware.tests import base
def test_start_transfer(self):
    data = b'image-data-here'
    read_handle = io.BytesIO(data)
    write_handle = mock.Mock()
    image_transfer._start_transfer(read_handle, write_handle, None)
    write_handle.write.assert_called_once_with(data)