from openstack.baremetal_introspection.v1 import _proxy
from openstack import service_description
class BaremetalIntrospectionService(service_description.ServiceDescription):
    """The bare metal introspection service."""
    supported_versions = {'1': _proxy.Proxy}