from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_remove_metadef_property(self):
    self.policy.remove_metadef_property()
    self.enforcer.enforce.assert_called_once_with(self.context, 'remove_metadef_property', mock.ANY)