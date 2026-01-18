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
def test_validate_with_ignorable_errors(self):
    t = template_format.parse('\n            heat_template_version: 2015-10-15\n            resources:\n              my_instance:\n                type: AWS::EC2::Instance\n            ')
    engine = service.EngineService('a', 't')
    self.mock_is_service_available.return_value = (False, 'Service endpoint not in service catalog.')
    res = dict(engine.validate_template(self.ctx, t, {}, ignorable_errors=[exception.ResourceTypeUnavailable.error_code]))
    expected = {'Description': 'No description', 'Parameters': {}, 'Environment': self.empty_environment}
    self.assertEqual(expected, res)