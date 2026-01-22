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
class TestEnforcer(base.BaseTestCase):

    def setUp(self):
        super(TestEnforcer, self).setUp()
        self.deltas = dict()
        self.config_fixture = self.useFixture(config_fixture.Config(CONF))
        self.config_fixture.config(group='oslo_limit', auth_type='password')
        self.config_fixture.config(group='oslo_limit', endpoint_id='ENDPOINT_ID')
        opts.register_opts(CONF)
        self.config_fixture.config(group='oslo_limit', auth_url='http://identity.example.com')
        limit._SDK_CONNECTION = mock.MagicMock()
        json = mock.MagicMock()
        json.json.return_value = {'model': {'name': 'flat'}}
        limit._SDK_CONNECTION.get.return_value = json

    def _get_usage_for_project(self, project_id, resource_names):
        return {'a': 1}

    def test_usage_callback_must_be_callable(self):
        invalid_callback_types = [uuid.uuid4().hex, 5, 5.1]
        for invalid_callback in invalid_callback_types:
            self.assertRaises(ValueError, limit.Enforcer, invalid_callback)

    def test_deltas_must_be_a_dictionary(self):
        project_id = uuid.uuid4().hex
        invalid_delta_types = [uuid.uuid4().hex, 5, 5.1, True, [], None, {}]
        enforcer = limit.Enforcer(self._get_usage_for_project)
        for invalid_delta in invalid_delta_types:
            self.assertRaises(ValueError, enforcer.enforce, project_id, invalid_delta)

    def test_project_id_must_be_a_string(self):
        enforcer = limit.Enforcer(self._get_usage_for_project)
        invalid_delta_types = [{}, 5, 5.1, True, False, [], None, '']
        for invalid_project_id in invalid_delta_types:
            self.assertRaises(ValueError, enforcer.enforce, invalid_project_id, {})

    def test_set_model_impl(self):
        enforcer = limit.Enforcer(self._get_usage_for_project)
        self.assertIsInstance(enforcer.model, limit._FlatEnforcer)

    def test_get_model_impl(self):
        json = mock.MagicMock()
        limit._SDK_CONNECTION.get.return_value = json
        json.json.return_value = {'model': {'name': 'flat'}}
        enforcer = limit.Enforcer(self._get_usage_for_project)
        flat_impl = enforcer._get_model_impl(self._get_usage_for_project)
        self.assertIsInstance(flat_impl, limit._FlatEnforcer)
        json.json.return_value = {'model': {'name': 'strict-two-level'}}
        flat_impl = enforcer._get_model_impl(self._get_usage_for_project)
        self.assertIsInstance(flat_impl, limit._StrictTwoLevelEnforcer)
        json.json.return_value = {'model': {'name': 'foo'}}
        e = self.assertRaises(ValueError, enforcer._get_model_impl, self._get_usage_for_project)
        self.assertEqual('enforcement model foo is not supported', str(e))

    @mock.patch.object(limit._FlatEnforcer, 'enforce')
    def test_enforce(self, mock_enforce):
        enforcer = limit.Enforcer(self._get_usage_for_project)
        project_id = uuid.uuid4().hex
        deltas = {'a': 1}
        enforcer.enforce(project_id, deltas)
        mock_enforce.assert_called_once_with(project_id, deltas)

    @mock.patch.object(limit._EnforcerUtils, 'get_project_limits')
    def test_calculate_usage(self, mock_get_limits):
        mock_usage = mock.MagicMock()
        mock_usage.return_value = {'a': 1, 'b': 2}
        project_id = uuid.uuid4().hex
        mock_get_limits.return_value = [('a', 10), ('b', 5)]
        expected = {'a': limit.ProjectUsage(10, 1), 'b': limit.ProjectUsage(5, 2)}
        enforcer = limit.Enforcer(mock_usage)
        self.assertEqual(expected, enforcer.calculate_usage(project_id, ['a', 'b']))

    @mock.patch.object(limit._EnforcerUtils, '_get_project_limit')
    @mock.patch.object(limit._EnforcerUtils, '_get_registered_limit')
    def test_calculate_and_enforce_some_missing(self, mock_get_reglimit, mock_get_limit):
        reg_limits = {'a': mock.MagicMock(default_limit=10), 'b': mock.MagicMock(default_limit=10)}
        prj_limits = {('bar', 'b'): mock.MagicMock(resource_limit=6)}
        mock_get_reglimit.side_effect = lambda r: reg_limits.get(r)
        mock_get_limit.side_effect = lambda p, r: prj_limits.get((p, r))
        mock_usage = mock.MagicMock()
        mock_usage.return_value = {'a': 5, 'b': 5, 'c': 5}
        enforcer = limit.Enforcer(mock_usage)
        expected = {'a': limit.ProjectUsage(10, 5), 'b': limit.ProjectUsage(6, 5), 'c': limit.ProjectUsage(0, 5)}
        self.assertEqual(expected, enforcer.calculate_usage('bar', ['a', 'b', 'c']))
        self.assertRaises(exception.ProjectOverLimit, enforcer.enforce, 'bar', {'a': 1, 'b': 0, 'c': 1})

    def test_calculate_usage_bad_params(self):
        enforcer = limit.Enforcer(mock.MagicMock())
        self.assertRaises(ValueError, enforcer.calculate_usage, 123, ['foo'])
        self.assertRaises(ValueError, enforcer.calculate_usage, 'project', [])
        self.assertRaises(ValueError, enforcer.calculate_usage, 'project', 123)
        self.assertRaises(ValueError, enforcer.calculate_usage, 'project', ['a', 123, 'b'])

    @mock.patch.object(limit._EnforcerUtils, 'get_registered_limits')
    def test_get_registered_limits(self, mock_get_limits):
        mock_get_limits.return_value = [('a', 1), ('b', 0), ('c', 2)]
        enforcer = limit.Enforcer(lambda: None)
        limits = enforcer.get_registered_limits(['a', 'b', 'c'])
        mock_get_limits.assert_called_once_with(['a', 'b', 'c'])
        self.assertEqual(mock_get_limits.return_value, limits)

    @mock.patch.object(limit._EnforcerUtils, 'get_project_limits')
    def test_get_project_limits(self, mock_get_limits):
        project_id = uuid.uuid4().hex
        mock_get_limits.return_value = [('a', 1), ('b', 0), ('c', 2)]
        enforcer = limit.Enforcer(lambda: None)
        limits = enforcer.get_project_limits(project_id, ['a', 'b', 'c'])
        mock_get_limits.assert_called_once_with(project_id, ['a', 'b', 'c'])
        self.assertEqual(mock_get_limits.return_value, limits)