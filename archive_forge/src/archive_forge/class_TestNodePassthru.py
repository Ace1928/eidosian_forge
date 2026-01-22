from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
@mock.patch.object(utils, 'pick_microversion', lambda session, v: v)
@mock.patch.object(node.Node, 'fetch', lambda self, session: self)
@mock.patch.object(exceptions, 'raise_from_response', mock.Mock())
class TestNodePassthru(object):

    def setUp(self):
        super(TestNodePassthru, self).setUp()
        self.node = node.Node(**FAKE)
        self.session = node.Mock(spec=adapter.Adapter, default_microversion='1.37')
        self.session.log = mock.Mock()

    def test_get_passthru(self):
        self.node.call_vendor_passthru(self.session, 'GET', 'test_method')
        self.session.get.assert_called_once_with('nodes/%s/vendor_passthru?method=test_method' % self.node.id, headers=mock.ANY, microversion='1.37', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_post_passthru(self):
        self.node.call_vendor_passthru(self.session, 'POST', 'test_method')
        self.session.post.assert_called_once_with('nodes/%s/vendor_passthru?method=test_method' % self.node.id, headers=mock.ANY, microversion='1.37', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_put_passthru(self):
        self.node.call_vendor_passthru(self.session, 'PUT', 'test_method')
        self.session.put.assert_called_once_with('nodes/%s/vendor_passthru?method=test_method' % self.node.id, headers=mock.ANY, microversion='1.37', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_delete_passthru(self):
        self.node.call_vendor_passthru(self.session, 'DELETE', 'test_method')
        self.session.delete.assert_called_once_with('nodes/%s/vendor_passthru?method=test_method' % self.node.id, headers=mock.ANY, microversion='1.37', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_list_passthru(self):
        self.node.list_vendor_passthru(self.session)
        self.session.get.assert_called_once_with('nodes/%s/vendor_passthru/methods' % self.node.id, headers=mock.ANY, microversion='1.37', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)