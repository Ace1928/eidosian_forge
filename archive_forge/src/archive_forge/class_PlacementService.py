from openstack.placement.v1 import _proxy
from openstack import service_description
class PlacementService(service_description.ServiceDescription):
    """The placement service."""
    supported_versions = {'1': _proxy.Proxy}