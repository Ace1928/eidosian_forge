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
def test_validate_redirect_pool_action_with_url(self):
    tmpl = yaml.safe_load(inline_templates.L7POLICY_TEMPLATE)
    props = tmpl['resources']['l7policy']['properties']
    props['action'] = 'REDIRECT_TO_POOL'
    props['redirect_pool'] = '123'
    self._create_stack(tmpl=yaml.safe_dump(tmpl))
    msg = _('redirect_url property should only be specified for action with value REDIRECT_TO_URL.')
    with mock.patch('heat.engine.clients.os.neutron.NeutronClientPlugin.has_extension', return_value=True):
        self.assertRaisesRegex(exception.ResourcePropertyValueDependency, msg, self.l7policy.validate)