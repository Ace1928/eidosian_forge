from unittest import mock
import testtools
from troveclient import base
from troveclient.v1 import instances
def test_restart_with_obj(self):
    self._test_restart(self.instance_with_id, self.instance_with_id.id)