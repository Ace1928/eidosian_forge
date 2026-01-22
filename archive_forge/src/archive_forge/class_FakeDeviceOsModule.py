import base64
import os
import sys
import mock
from pyu2f.hid import linux
class FakeDeviceOsModule(object):
    O_RDWR = os.O_RDWR
    path = os.path
    data_written = None
    data_to_return = None

    def open(self, unused_path, unused_opts):
        return 0

    def write(self, unused_dev, data):
        self.data_written = data

    def read(self, unused_dev, unused_length):
        return self.data_to_return