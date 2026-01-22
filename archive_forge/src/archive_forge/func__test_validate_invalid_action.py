from unittest import mock
import yaml
from neutronclient.common import exceptions
from heat.common import exception
from heat.common import template_format
from heat.tests import common
from heat.tests.openstack.neutron import inline_templates
from heat.tests import utils
def _test_validate_invalid_action(self, msg, invalid_action='invalid', obj_type='network'):
    tpl = yaml.safe_load(inline_templates.RBAC_TEMPLATE)
    tpl['resources']['rbac']['properties']['action'] = invalid_action
    tpl['resources']['rbac']['properties']['object_type'] = obj_type
    self._create_stack(tmpl=yaml.safe_dump(tpl))
    self.patchobject(type(self.rbac), 'is_service_available', return_value=(True, None))
    self.assertRaisesRegex(exception.StackValidationFailed, msg, self.rbac.validate)