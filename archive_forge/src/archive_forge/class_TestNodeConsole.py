from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
@mock.patch.object(node.Node, 'fetch', lambda self, session: self)
@mock.patch.object(exceptions, 'raise_from_response', mock.Mock())
class TestNodeConsole(base.TestCase):

    def setUp(self):
        super().setUp()
        self.node = node.Node(**FAKE)
        self.session = mock.Mock(spec=adapter.Adapter, default_microversion='1.1')

    def test_get_console(self):
        self.node.get_console(self.session)
        self.session.get.assert_called_once_with('nodes/%s/states/console' % self.node.id, headers=mock.ANY, microversion=mock.ANY, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_set_console_mode(self):
        self.node.set_console_mode(self.session, True)
        self.session.put.assert_called_once_with('nodes/%s/states/console' % self.node.id, json={'enabled': True}, headers=mock.ANY, microversion=mock.ANY, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_set_console_mode_invalid_enabled(self):
        self.assertRaises(ValueError, self.node.set_console_mode, self.session, 'true')