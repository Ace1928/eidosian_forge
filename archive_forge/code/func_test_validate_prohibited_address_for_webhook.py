from unittest import mock
from heat.common import exception
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.clients.os import monasca as client_plugin
from heat.engine.resources.openstack.monasca import notification
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_validate_prohibited_address_for_webhook(self):
    self.test_resource.properties.data['type'] = self.test_resource.WEBHOOK
    self.test_resource.properties.data['address'] = 'ftp://127.0.0.1'
    ex = self.assertRaises(exception.StackValidationFailed, self.test_resource.validate)
    self.assertEqual('Address "ftp://127.0.0.1" doesn\'t satisfies allowed schemes: http, https', str(ex))