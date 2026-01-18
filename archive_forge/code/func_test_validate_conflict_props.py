from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.neutron import security_group_rule
from heat.tests import common
from heat.tests.openstack.neutron import inline_templates
from heat.tests import utils
def test_validate_conflict_props(self):
    self.patchobject(security_group_rule.SecurityGroupRule, 'is_service_available', return_value=(True, None))
    tmpl = inline_templates.SECURITY_GROUP_RULE_TEMPLATE
    tmpl += '      remote_ip_prefix: "10.0.0.0/8"'
    self._create_stack(tmpl=tmpl)
    self.assertRaises(exception.ResourcePropertyConflict, self.sg_rule.validate)