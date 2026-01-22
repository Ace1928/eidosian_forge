from unittest import mock
import testtools
from troveclient import base
from troveclient.v1 import instances
def _test_resize_instance(self, instance, id):
    self._set_action_mock()
    self.instances.resize_instance(instance, 103)
    self.assertEqual(id, self._instance_id)
    self.assertEqual({'resize': {'flavorRef': 103}}, self._body)