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
def test_validate_hot_no_parameter_label(self):
    t = template_format.parse(test_template_hot_no_parameter_label)
    res = dict(self.engine.validate_template(self.ctx, t, {}))
    parameters = res['Parameters']
    expected = {'KeyName': {'Type': 'String', 'Description': 'Name of an existing key pair to use for the instance', 'NoEcho': 'false', 'Label': 'KeyName'}}
    self.assertEqual(expected, parameters)