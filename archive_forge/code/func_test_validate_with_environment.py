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
def test_validate_with_environment(self):
    test_template = test_template_ref % 'WikiDatabase'
    test_template = test_template.replace('AWS::EC2::Instance', 'My::Instance')
    t = template_format.parse(test_template)
    params = {'resource_registry': {'My::Instance': 'AWS::EC2::Instance'}}
    res = dict(self.engine.validate_template(self.ctx, t, params))
    self.assertEqual('test.', res['Description'])