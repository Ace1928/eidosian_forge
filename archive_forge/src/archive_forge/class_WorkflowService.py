from openstack import service_description
from openstack.workflow.v2 import _proxy
class WorkflowService(service_description.ServiceDescription):
    """The workflow service."""
    supported_versions = {'2': _proxy.Proxy}