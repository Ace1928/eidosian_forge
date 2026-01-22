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
class TestProxyCleanup(base.TestCase):

    def setUp(self):
        super(TestProxyCleanup, self).setUp()
        self.session = mock.Mock()
        self.session._sdk_connection = self.cloud
        self.fake_id = 1
        self.fake_name = 'fake_name'
        self.fake_result = 'fake_result'
        self.res = mock.Mock(spec=resource.Resource)
        self.res.id = self.fake_id
        self.res.created_at = '2020-01-02T03:04:05'
        self.res.updated_at = '2020-01-03T03:04:05'
        self.res_no_updated = mock.Mock(spec=resource.Resource)
        self.res_no_updated.created_at = '2020-01-02T03:04:05'
        self.sot = proxy.Proxy(self.session)
        self.sot.service_type = 'block-storage'
        self.delete_mock = mock.Mock()

    def test_filters_evaluation_created_at(self):
        self.assertTrue(self.sot._service_cleanup_resource_filters_evaluation(self.res, filters={'created_at': '2020-02-03T00:00:00'}))

    def test_filters_evaluation_created_at_not(self):
        self.assertFalse(self.sot._service_cleanup_resource_filters_evaluation(self.res, filters={'created_at': '2020-01-01T00:00:00'}))

    def test_filters_evaluation_updated_at(self):
        self.assertTrue(self.sot._service_cleanup_resource_filters_evaluation(self.res, filters={'updated_at': '2020-02-03T00:00:00'}))

    def test_filters_evaluation_updated_at_not(self):
        self.assertFalse(self.sot._service_cleanup_resource_filters_evaluation(self.res, filters={'updated_at': '2020-01-01T00:00:00'}))

    def test_filters_evaluation_updated_at_missing(self):
        self.assertFalse(self.sot._service_cleanup_resource_filters_evaluation(self.res_no_updated, filters={'updated_at': '2020-01-01T00:00:00'}))

    def test_filters_empty(self):
        self.assertTrue(self.sot._service_cleanup_resource_filters_evaluation(self.res_no_updated))

    def test_service_cleanup_dry_run(self):
        self.assertTrue(self.sot._service_cleanup_del_res(self.delete_mock, self.res, dry_run=True))
        self.delete_mock.assert_not_called()

    def test_service_cleanup_dry_run_default(self):
        self.assertTrue(self.sot._service_cleanup_del_res(self.delete_mock, self.res))
        self.delete_mock.assert_not_called()

    def test_service_cleanup_real_run(self):
        self.assertTrue(self.sot._service_cleanup_del_res(self.delete_mock, self.res, dry_run=False))
        self.delete_mock.assert_called_with(self.res)

    def test_service_cleanup_real_run_identified_resources(self):
        rd = dict()
        self.assertTrue(self.sot._service_cleanup_del_res(self.delete_mock, self.res, dry_run=False, identified_resources=rd))
        self.delete_mock.assert_called_with(self.res)
        self.assertEqual(self.res, rd[self.res.id])

    def test_service_cleanup_resource_evaluation_false(self):
        self.assertFalse(self.sot._service_cleanup_del_res(self.delete_mock, self.res, dry_run=False, resource_evaluation_fn=lambda x, y, z: False))
        self.delete_mock.assert_not_called()

    def test_service_cleanup_resource_evaluation_true(self):
        self.assertTrue(self.sot._service_cleanup_del_res(self.delete_mock, self.res, dry_run=False, resource_evaluation_fn=lambda x, y, z: True))
        self.delete_mock.assert_called()

    def test_service_cleanup_resource_evaluation_override_filters(self):
        self.assertFalse(self.sot._service_cleanup_del_res(self.delete_mock, self.res, dry_run=False, resource_evaluation_fn=lambda x, y, z: False, filters={'created_at': '2200-01-01'}))

    def test_service_cleanup_filters(self):
        self.assertTrue(self.sot._service_cleanup_del_res(self.delete_mock, self.res, dry_run=False, filters={'created_at': '2200-01-01'}))
        self.delete_mock.assert_called()

    def test_service_cleanup_queue(self):
        q = queue.Queue()
        self.assertTrue(self.sot._service_cleanup_del_res(self.delete_mock, self.res, dry_run=False, client_status_queue=q, filters={'created_at': '2200-01-01'}))
        self.assertEqual(self.res, q.get_nowait())

    def test_should_skip_resource_cleanup(self):
        excluded = ['block_storage.backup']
        self.assertTrue(self.sot.should_skip_resource_cleanup('backup', excluded))
        self.assertFalse(self.sot.should_skip_resource_cleanup('volume', excluded))