from openstack.dns.v2 import _proxy
from openstack import service_description
class DnsService(service_description.ServiceDescription):
    """The DNS service."""
    supported_versions = {'2': _proxy.Proxy}