from openstack import proxy
from openstack.workflow.v2 import cron_trigger as _cron_trigger
from openstack.workflow.v2 import execution as _execution
from openstack.workflow.v2 import workflow as _workflow
def update_workflow(self, workflow, **attrs):
    """Update workflow from attributes

        :param workflow: The value can be either the name of a workflow or a
            :class:`~openstack.workflow.v2.workflow.Workflow`
            instance.
        :param dict attrs: Keyword arguments which will be used to update
            a :class:`~openstack.workflow.v2.workflow.Workflow`,
            comprised of the properties on the Workflow class.

        :returns: The results of workflow update
        :rtype: :class:`~openstack.workflow.v2.workflow.Workflow`
        """
    return self._update(_workflow.Workflow, workflow, **attrs)