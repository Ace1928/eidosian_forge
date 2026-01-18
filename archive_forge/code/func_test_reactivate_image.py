from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_reactivate_image(self):
    self.policy.reactivate_image()
    self.enforcer.enforce.assert_called_once_with(self.context, 'reactivate', mock.ANY)