from unittest import mock
import uuid
from openstack.identity.v3 import endpoint
from openstack.identity.v3 import limit as klimit
from openstack.identity.v3 import registered_limit
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslotest import base
from oslo_limit import exception
from oslo_limit import fixture
from oslo_limit import limit
from oslo_limit import opts
class TestFlatEnforcer(base.BaseTestCase):

    def setUp(self):
        super(TestFlatEnforcer, self).setUp()
        self.config_fixture = self.useFixture(config_fixture.Config(CONF))
        self.config_fixture.config(group='oslo_limit', endpoint_id='ENDPOINT_ID')
        opts.register_opts(CONF)
        self.mock_conn = mock.MagicMock()
        limit._SDK_CONNECTION = self.mock_conn

    @mock.patch.object(limit._EnforcerUtils, 'get_registered_limits')
    def test_get_registered_limits(self, mock_get_limits):
        mock_get_limits.return_value = [('a', 1), ('b', 0), ('c', 2)]
        enforcer = limit._FlatEnforcer(lambda: None)
        limits = enforcer.get_registered_limits(['a', 'b', 'c'])
        mock_get_limits.assert_called_once_with(['a', 'b', 'c'])
        self.assertEqual(mock_get_limits.return_value, limits)

    @mock.patch.object(limit._EnforcerUtils, 'get_project_limits')
    def test_get_project_limits(self, mock_get_limits):
        project_id = uuid.uuid4().hex
        mock_get_limits.return_value = [('a', 1), ('b', 0), ('c', 2)]
        enforcer = limit._FlatEnforcer(lambda: None)
        limits = enforcer.get_project_limits(project_id, ['a', 'b', 'c'])
        mock_get_limits.assert_called_once_with(project_id, ['a', 'b', 'c'])
        self.assertEqual(mock_get_limits.return_value, limits)

    @mock.patch.object(limit._EnforcerUtils, 'get_project_limits')
    def test_enforce(self, mock_get_limits):
        mock_usage = mock.MagicMock()
        project_id = uuid.uuid4().hex
        deltas = {'a': 1, 'b': 1}
        mock_get_limits.return_value = [('a', 1), ('b', 2)]
        mock_usage.return_value = {'a': 0, 'b': 1}
        enforcer = limit._FlatEnforcer(mock_usage)
        enforcer.enforce(project_id, deltas)
        self.mock_conn.get_endpoint.assert_called_once_with('ENDPOINT_ID')
        mock_get_limits.assert_called_once_with(project_id, ['a', 'b'])
        mock_usage.assert_called_once_with(project_id, ['a', 'b'])

    @mock.patch.object(limit._EnforcerUtils, 'get_project_limits')
    def test_enforce_raises_on_over(self, mock_get_limits):
        mock_usage = mock.MagicMock()
        project_id = uuid.uuid4().hex
        deltas = {'a': 2, 'b': 1}
        mock_get_limits.return_value = [('a', 1), ('b', 2)]
        mock_usage.return_value = {'a': 0, 'b': 1}
        enforcer = limit._FlatEnforcer(mock_usage)
        e = self.assertRaises(exception.ProjectOverLimit, enforcer.enforce, project_id, deltas)
        expected = 'Project %s is over a limit for [Resource a is over limit of 1 due to current usage 0 and delta 2]'
        self.assertEqual(expected % project_id, str(e))
        self.assertEqual(project_id, e.project_id)
        self.assertEqual(1, len(e.over_limit_info_list))
        over_a = e.over_limit_info_list[0]
        self.assertEqual('a', over_a.resource_name)
        self.assertEqual(1, over_a.limit)
        self.assertEqual(0, over_a.current_usage)
        self.assertEqual(2, over_a.delta)

    @mock.patch.object(limit._EnforcerUtils, '_get_project_limit')
    @mock.patch.object(limit._EnforcerUtils, '_get_registered_limit')
    def test_enforce_raises_on_missing_limit(self, mock_get_reglimit, mock_get_limit):

        def mock_usage(*a):
            return {'a': 1, 'b': 1}
        project_id = uuid.uuid4().hex
        deltas = {'a': 0, 'b': 0}
        mock_get_reglimit.return_value = None
        mock_get_limit.return_value = None
        enforcer = limit._FlatEnforcer(mock_usage)
        self.assertRaises(exception.ProjectOverLimit, enforcer.enforce, project_id, deltas)
        self.assertRaises(exception.ProjectOverLimit, enforcer.enforce, None, deltas)