from unittest import mock
import testtools
from troveclient import base
from troveclient.v1 import instances
def _set_action_mock(self):

    def side_effect_func(instance, body):
        self._instance_id = base.getid(instance)
        self._body = body
    self._instance_id = None
    self._body = None
    self.instances._action = mock.Mock(side_effect=side_effect_func)