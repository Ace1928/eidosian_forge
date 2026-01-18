from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_update_property(self):
    with mock.patch.object(self.policy, '_enforce') as mock_enf:
        self.policy.update_property('foo', None)
        mock_enf.assert_called_once_with('modify_image')
    with mock.patch.object(self.policy, '_enforce_visibility') as mock_enf:
        self.policy.update_property('visibility', 'foo')
        mock_enf.assert_called_once_with('foo')