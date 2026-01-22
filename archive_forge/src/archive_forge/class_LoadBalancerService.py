from openstack.load_balancer.v2 import _proxy
from openstack import service_description
class LoadBalancerService(service_description.ServiceDescription):
    """The load balancer service."""
    supported_versions = {'2': _proxy.Proxy}