import os
import re
import base64
import collections
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey, KeyCertificateConnection
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import StorageVolume
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.container.providers import Provider
def _to_image(self, metadata):
    """
        Returns a container image from the given metadata

        :param metadata:
        :type  metadata: ``dict``

        :rtype: :class:`.ContainerImage`
        """
    fingerprint = metadata.get('fingerprint')
    aliases = metadata.get('aliases', [])
    if aliases:
        name = metadata.get('aliases')[0].get('name')
    else:
        name = metadata.get('properties', {}).get('description') or fingerprint
    version = metadata.get('update_source', {}).get('alias')
    extra = metadata
    return ContainerImage(id=fingerprint, name=name, path=None, version=version, driver=self, extra=extra)