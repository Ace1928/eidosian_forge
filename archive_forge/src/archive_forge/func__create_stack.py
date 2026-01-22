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
def _create_stack(self, tmpl=inline_templates.L7POLICY_TEMPLATE):
    self.t = template_format.parse(tmpl)
    self.stack = utils.parse_stack(self.t)
    self.l7policy = self.stack['l7policy']
    self.octavia_client = mock.MagicMock()
    self.l7policy.client = mock.MagicMock(return_value=self.octavia_client)
    self.l7policy.client_plugin().client = mock.MagicMock(return_value=self.octavia_client)