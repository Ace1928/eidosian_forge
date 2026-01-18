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
def test_update_redirect_pool_prop_name(self):
    self._create_stack()
    self.l7policy.resource_id_set('1234')
    self.octavia_client.l7policy_show.side_effect = [{'provisioning_status': 'PENDING_UPDATE'}, {'provisioning_status': 'PENDING_UPDATE'}, {'provisioning_status': 'ACTIVE'}]
    self.octavia_client.l7policy_set.side_effect = [exceptions.Conflict(409), None]
    unresolved_diff = {'redirect_url': None, 'action': 'REDIRECT_TO_POOL', 'redirect_pool': 'UNRESOLVED_POOL'}
    resolved_diff = {'redirect_url': None, 'action': 'REDIRECT_TO_POOL', 'redirect_pool_id': '123'}
    self.l7policy.handle_update(None, None, unresolved_diff)
    self.assertFalse(self.l7policy.check_update_complete(resolved_diff))
    self.assertFalse(self.l7policy._update_called)
    self.octavia_client.l7policy_set.assert_called_with('1234', json={'l7policy': resolved_diff})
    self.assertFalse(self.l7policy.check_update_complete(resolved_diff))
    self.assertTrue(self.l7policy._update_called)
    self.octavia_client.l7policy_set.assert_called_with('1234', json={'l7policy': resolved_diff})
    self.assertFalse(self.l7policy.check_update_complete(resolved_diff))
    self.assertTrue(self.l7policy.check_update_complete(resolved_diff))