from openstack.tests.unit import test_proxy_base
from openstack.workflow.v2 import _proxy
from openstack.workflow.v2 import cron_trigger
from openstack.workflow.v2 import execution
from openstack.workflow.v2 import workflow
class TestWorkflowProxy(test_proxy_base.TestProxyBase):

    def setUp(self):
        super(TestWorkflowProxy, self).setUp()
        self.proxy = _proxy.Proxy(self.session)

    def test_workflows(self):
        self.verify_list(self.proxy.workflows, workflow.Workflow)

    def test_executions(self):
        self.verify_list(self.proxy.executions, execution.Execution)

    def test_workflow_get(self):
        self.verify_get(self.proxy.get_workflow, workflow.Workflow)

    def test_execution_get(self):
        self.verify_get(self.proxy.get_execution, execution.Execution)

    def test_workflow_create(self):
        self.verify_create(self.proxy.create_workflow, workflow.Workflow)

    def test_workflow_update(self):
        self.verify_update(self.proxy.update_workflow, workflow.Workflow)

    def test_execution_create(self):
        self.verify_create(self.proxy.create_execution, execution.Execution)

    def test_workflow_delete(self):
        self.verify_delete(self.proxy.delete_workflow, workflow.Workflow, True)

    def test_execution_delete(self):
        self.verify_delete(self.proxy.delete_execution, execution.Execution, True)

    def test_workflow_find(self):
        self.verify_find(self.proxy.find_workflow, workflow.Workflow)

    def test_execution_find(self):
        self.verify_find(self.proxy.find_execution, execution.Execution)