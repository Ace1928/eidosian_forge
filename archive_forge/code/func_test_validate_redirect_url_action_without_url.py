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
def test_validate_redirect_url_action_without_url(self):
    tmpl = yaml.safe_load(inline_templates.L7POLICY_TEMPLATE)
    props = tmpl['resources']['l7policy']['properties']
    del props['redirect_url']
    self._create_stack(tmpl=yaml.safe_dump(tmpl))
    msg = _('Property redirect_url is required when action type is set to REDIRECT_TO_URL.')
    with mock.patch('heat.engine.clients.os.neutron.NeutronClientPlugin.has_extension', return_value=True):
        self.assertRaisesRegex(exception.StackValidationFailed, msg, self.l7policy.validate)