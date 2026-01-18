from unittest import mock
from webob import exc
from heat.api.openstack.v1 import util
from heat.common import context
from heat.common import policy
from heat.common import wsgi
from heat.tests import common
def test_returns_only_allowed_params(self):
    self.params.add('bar', 'bar value')
    result = util.get_allowed_params(self.params, self.param_types)
    self.assertIn('foo', result)
    self.assertNotIn('bar', result)