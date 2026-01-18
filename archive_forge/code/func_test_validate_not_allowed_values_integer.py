from unittest import mock
from oslo_messaging.rpc import dispatcher
import webob
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.common import urlfetch
from heat.engine.clients.os import glance
from heat.engine import environment
from heat.engine.hot import template as hot_tmpl
from heat.engine import resources
from heat.engine import service
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_validate_not_allowed_values_integer(self):
    t = template_format.parse(test_template_allowed_integers)
    template = tmpl.Template(t, env=environment.Environment({'size': '3'}))
    stack = parser.Stack(self.ctx, 'test_stack', template)
    err = self.assertRaises(exception.StackValidationFailed, stack.validate)
    self.assertIn('"3" is not an allowed value [1, 4, 8]', str(err))
    template.env = environment.Environment({'size': 3})
    stack = parser.Stack(self.ctx, 'test_stack', template)
    err = self.assertRaises(exception.StackValidationFailed, stack.validate)
    self.assertIn('3 is not an allowed value [1, 4, 8]', str(err))