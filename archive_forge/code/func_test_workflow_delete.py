from openstack.tests.unit import test_proxy_base
from openstack.workflow.v2 import _proxy
from openstack.workflow.v2 import cron_trigger
from openstack.workflow.v2 import execution
from openstack.workflow.v2 import workflow
def test_workflow_delete(self):
    self.verify_delete(self.proxy.delete_workflow, workflow.Workflow, True)