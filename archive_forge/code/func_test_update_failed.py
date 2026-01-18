import collections
import json
from unittest import mock
from heatclient import exc
from heatclient.v1 import stacks
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.common import policy
from heat.common import template_format
from heat.engine.clients.os import heat_plugin
from heat.engine import environment
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.heat import remote_stack
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.tests import common as tests_common
from heat.tests import utils
def test_update_failed(self):
    stacks = [get_stack(stack_status='UPDATE_IN_PROGRESS'), get_stack(stack_status='UPDATE_FAILED', stack_status_reason='Remote stack update failed')]
    rsrc = self.create_remote_stack()
    props = dict(rsrc.properties)
    props['parameters']['name'] = 'bar'
    update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    self.heat.stacks.get = mock.MagicMock(side_effect=stacks)
    error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.update, update_snippet))
    error_msg = _('ResourceInError: resources.remote_stack: Went to status UPDATE_FAILED due to "Remote stack update failed"')
    self.assertEqual(error_msg, str(error))
    self.assertEqual((rsrc.UPDATE, rsrc.FAILED), rsrc.state)
    self.assertEqual(2, len(self.heat.stacks.get.call_args_list))