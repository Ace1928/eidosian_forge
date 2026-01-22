from unittest import mock
from webob import exc
from heat.api.openstack.v1 import util
from heat.common import context
from heat.common import policy
from heat.common import wsgi
from heat.tests import common
class TestGetAllowedParams(common.HeatTestCase):

    def setUp(self):
        super(TestGetAllowedParams, self).setUp()
        req = wsgi.Request({})
        self.params = req.params.copy()
        self.params.add('foo', 'foo value')
        self.param_types = {'foo': util.PARAM_TYPE_SINGLE}

    def test_returns_empty_dict(self):
        self.param_types = {}
        result = util.get_allowed_params(self.params, self.param_types)
        self.assertEqual({}, result)

    def test_only_adds_allowed_param_if_param_exists(self):
        self.param_types = {'foo': util.PARAM_TYPE_SINGLE}
        self.params.clear()
        result = util.get_allowed_params(self.params, self.param_types)
        self.assertNotIn('foo', result)

    def test_returns_only_allowed_params(self):
        self.params.add('bar', 'bar value')
        result = util.get_allowed_params(self.params, self.param_types)
        self.assertIn('foo', result)
        self.assertNotIn('bar', result)

    def test_handles_single_value_params(self):
        result = util.get_allowed_params(self.params, self.param_types)
        self.assertEqual('foo value', result['foo'])

    def test_handles_multiple_value_params(self):
        self.param_types = {'foo': util.PARAM_TYPE_MULTI}
        self.params.add('foo', 'foo value 2')
        result = util.get_allowed_params(self.params, self.param_types)
        self.assertEqual(2, len(result['foo']))
        self.assertIn('foo value', result['foo'])
        self.assertIn('foo value 2', result['foo'])

    def test_handles_mixed_value_param_with_multiple_entries(self):
        self.param_types = {'foo': util.PARAM_TYPE_MIXED}
        self.params.add('foo', 'foo value 2')
        result = util.get_allowed_params(self.params, self.param_types)
        self.assertEqual(2, len(result['foo']))
        self.assertIn('foo value', result['foo'])
        self.assertIn('foo value 2', result['foo'])

    def test_handles_mixed_value_param_with_single_entry(self):
        self.param_types = {'foo': util.PARAM_TYPE_MIXED}
        result = util.get_allowed_params(self.params, self.param_types)
        self.assertEqual('foo value', result['foo'])

    def test_bogus_param_type(self):
        self.param_types = {'foo': 'blah'}
        self.assertRaises(AssertionError, util.get_allowed_params, self.params, self.param_types)