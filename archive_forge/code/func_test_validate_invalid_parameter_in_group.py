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
def test_validate_invalid_parameter_in_group(self):
    t = template_format.parse(test_template_invalid_parameter_name)
    template = hot_tmpl.HOTemplate20130523(t, env=environment.Environment({'KeyName': 'test', 'ImageId': 'sometestid', 'db_password': 'Pass123'}))
    stack = parser.Stack(self.ctx, 'test_stack', template)
    exc = self.assertRaises(exception.StackValidationFailed, stack.validate)
    self.assertEqual(_('Parameter Groups error: parameter_groups.Database Group: The grouped parameter SomethingNotHere does not reference a valid parameter.'), str(exc))