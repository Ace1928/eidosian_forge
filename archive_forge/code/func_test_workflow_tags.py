from unittest import mock
import yaml
from mistralclient.api import base as mistral_base
from mistralclient.api.v2 import executions
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import mistral as client
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.mistral import workflow
from heat.engine.resources import signal_responder
from heat.engine.resources import stack_user
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_workflow_tags(self):
    tmpl = template_format.parse(workflow_template_with_tags)
    stack = utils.parse_stack(tmpl)
    rsrc_defns = stack.t.resource_definitions(stack)['workflow']
    wf = workflow.Workflow('workflow', rsrc_defns, stack)
    self.mistral.workflows.create.return_value = [FakeWorkflow('workflow')]
    scheduler.TaskRunner(wf.create)()
    details = {'tags': ['mytag'], 'params': {'test': 'param_value', 'test1': 'param_value_1'}}
    execution = mock.Mock()
    execution.id = '12345'
    self.mistral.executions.create.side_effect = lambda *args, **kw: self.verify_params(*args, **kw)
    scheduler.TaskRunner(wf.signal, details)()