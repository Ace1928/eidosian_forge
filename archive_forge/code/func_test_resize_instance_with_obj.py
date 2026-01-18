from unittest import mock
import testtools
from troveclient import base
from troveclient.v1 import instances
def test_resize_instance_with_obj(self):
    self._test_resize_instance(self.instance_with_id, self.instance_with_id.id)