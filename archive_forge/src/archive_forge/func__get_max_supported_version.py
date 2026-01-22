import logging
from keystoneauth1 import discover
from keystoneauth1.exceptions.http import NotAcceptable
from barbicanclient import client as base_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import cas
from barbicanclient.v1 import containers
from barbicanclient.v1 import orders
from barbicanclient.v1 import secrets
def _get_max_supported_version(self, session, endpoint, version, service_type, service_name, interface, region_name, microversion):
    min_ver, max_ver = self._get_min_max_server_supported_microversion(session, endpoint, version, service_type, service_name, interface, region_name)
    if microversion is None:
        for client_version in _SUPPORTED_MICROVERSIONS[::-1]:
            if discover.version_between(min_ver, max_ver, client_version):
                return self._get_normalized_microversion(client_version)
        raise ValueError("Couldn't find a version supported by both client and server")
    if discover.version_between(min_ver, max_ver, microversion):
        return microversion
    raise ValueError('Invalid microversion {}: Microversion requested is not supported by the server'.format(microversion))