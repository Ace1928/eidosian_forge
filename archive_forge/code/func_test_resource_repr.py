from unittest import mock
from oslotest import base as test_base
from ironicclient.common.apiclient import base
def test_resource_repr(self):
    r = base.Resource(None, dict(foo='bar', baz='spam'))
    self.assertEqual('<Resource baz=spam, foo=bar>', repr(r))