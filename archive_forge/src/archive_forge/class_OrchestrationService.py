from openstack.orchestration.v1 import _proxy
from openstack import service_description
class OrchestrationService(service_description.ServiceDescription):
    """The orchestration service."""
    supported_versions = {'1': _proxy.Proxy}