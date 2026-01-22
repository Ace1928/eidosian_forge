import copy
import queue
from unittest import mock
from keystoneauth1 import session
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack import proxy
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
class TestProxyHead(base.TestCase):

    def setUp(self):
        super(TestProxyHead, self).setUp()
        self.session = mock.Mock()
        self.session._sdk_connection = self.cloud
        self.fake_id = 1
        self.fake_name = 'fake_name'
        self.fake_result = 'fake_result'
        self.res = mock.Mock(spec=HeadableResource)
        self.res.id = self.fake_id
        self.res.head = mock.Mock(return_value=self.fake_result)
        self.sot = proxy.Proxy(self.session)
        self.sot._connection = self.cloud
        HeadableResource.new = mock.Mock(return_value=self.res)

    def test_head_resource(self):
        rv = self.sot._head(HeadableResource, self.res)
        self.res.head.assert_called_with(self.sot, base_path=None)
        self.assertEqual(rv, self.fake_result)

    def test_head_resource_base_path(self):
        base_path = 'dummy'
        rv = self.sot._head(HeadableResource, self.res, base_path=base_path)
        self.res.head.assert_called_with(self.sot, base_path=base_path)
        self.assertEqual(rv, self.fake_result)

    def test_head_id(self):
        rv = self.sot._head(HeadableResource, self.fake_id)
        HeadableResource.new.assert_called_with(connection=self.cloud, id=self.fake_id)
        self.res.head.assert_called_with(self.sot, base_path=None)
        self.assertEqual(rv, self.fake_result)