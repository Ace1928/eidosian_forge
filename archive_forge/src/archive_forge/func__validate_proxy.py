import typing as ty
import warnings
import os_service_types
from openstack import _log
from openstack import exceptions
from openstack import proxy as proxy_mod
from openstack import warnings as os_warnings
def _validate_proxy(self, proxy, endpoint):
    exc = None
    service_url = getattr(proxy, 'skip_discovery', None)
    try:
        if service_url is None:
            service_url = proxy.get_endpoint_data().service_url
    except Exception as e:
        exc = e
    if exc or not endpoint or (not service_url):
        raise exceptions.ServiceDiscoveryException('Failed to create a working proxy for service {service_type}: {message}'.format(service_type=self.service_type, message=exc or 'No valid endpoint was discoverable.'))