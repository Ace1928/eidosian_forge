from unittest import mock
import yaml
from osc_lib import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine.resources.openstack.octavia import l7policy
from heat.tests import common
from heat.tests.openstack.octavia import inline_templates
from heat.tests import utils
def test_validate_redirect_pool_action_without_pool(self):
    tmpl = yaml.safe_load(inline_templates.L7POLICY_TEMPLATE)
    props = tmpl['resources']['l7policy']['properties']
    props['action'] = 'REDIRECT_TO_POOL'
    del props['redirect_url']
    self._create_stack(tmpl=yaml.safe_dump(tmpl))
    msg = _('Property redirect_pool is required when action type is set to REDIRECT_TO_POOL.')
    with mock.patch('heat.engine.clients.os.neutron.NeutronClientPlugin.has_extension', return_value=True):
        self.assertRaisesRegex(exception.StackValidationFailed, msg, self.l7policy.validate)