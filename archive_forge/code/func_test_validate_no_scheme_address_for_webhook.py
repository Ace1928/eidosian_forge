from unittest import mock
from heat.common import exception
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.clients.os import monasca as client_plugin
from heat.engine.resources.openstack.monasca import notification
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_validate_no_scheme_address_for_webhook(self):
    self.test_resource.properties.data['type'] = self.test_resource.WEBHOOK
    self.test_resource.properties.data['address'] = 'abc@def.com'
    ex = self.assertRaises(exception.StackValidationFailed, self.test_resource.validate)
    self.assertEqual('Address "abc@def.com" doesn\'t have required URL scheme', str(ex))