from openstack.identity.v2 import _proxy as _proxy_v2
from openstack.identity.v3 import _proxy as _proxy_v3
from openstack import service_description
class IdentityService(service_description.ServiceDescription):
    """The identity service."""
    supported_versions = {'2': _proxy_v2.Proxy, '3': _proxy_v3.Proxy}