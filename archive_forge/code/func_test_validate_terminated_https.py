from unittest import mock
import yaml
from osc_lib import exceptions
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.octavia import listener
from heat.tests import common
from heat.tests.openstack.octavia import inline_templates
from heat.tests import utils
def test_validate_terminated_https(self):
    tmpl = yaml.safe_load(inline_templates.LISTENER_TEMPLATE)
    props = tmpl['resources']['listener']['properties']
    props['protocol'] = 'TERMINATED_HTTPS'
    del props['default_tls_container_ref']
    self._create_stack(tmpl=yaml.safe_dump(tmpl))
    self.assertRaises(exception.StackValidationFailed, self.listener.validate)