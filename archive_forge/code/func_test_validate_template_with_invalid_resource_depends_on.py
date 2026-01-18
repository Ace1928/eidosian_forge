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
def test_validate_template_with_invalid_resource_depends_on(self):
    hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        resources:\n          resource1:\n            type: AWS::EC2::Instance\n            properties:\n              property1: value1\n            metadata:\n              foo: bar\n            DependsOn: dummy\n            deletion_policy: dummy\n            update_policy:\n              foo: bar\n        ')
    res = dict(self.engine.validate_template(self.ctx, hot_tpl, {}))
    self.assertEqual({'Error': '"DependsOn" is not a valid keyword inside a resource definition'}, res)