import logging
from keystoneauth1 import discover
from keystoneauth1.exceptions.http import NotAcceptable
from barbicanclient import client as base_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import cas
from barbicanclient.v1 import containers
from barbicanclient.v1 import orders
from barbicanclient.v1 import secrets
def _get_normalized_microversion(self, microversion):
    if microversion is None:
        return
    normalized = discover.normalize_version_number(microversion)
    if normalized not in _SUPPORTED_MICROVERSIONS:
        raise ValueError('Invalid microversion {}: Microversion requested is not supported by the client'.format(microversion))
    return discover.version_to_string(normalized)