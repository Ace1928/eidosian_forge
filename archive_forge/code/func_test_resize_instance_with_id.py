from unittest import mock
import testtools
from troveclient import base
from troveclient.v1 import instances
def test_resize_instance_with_id(self):
    self._test_resize_instance(4725, 4725)