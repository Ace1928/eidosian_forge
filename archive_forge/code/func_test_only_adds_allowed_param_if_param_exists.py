from unittest import mock
from webob import exc
from heat.api.openstack.v1 import util
from heat.common import context
from heat.common import policy
from heat.common import wsgi
from heat.tests import common
def test_only_adds_allowed_param_if_param_exists(self):
    self.param_types = {'foo': util.PARAM_TYPE_SINGLE}
    self.params.clear()
    result = util.get_allowed_params(self.params, self.param_types)
    self.assertNotIn('foo', result)