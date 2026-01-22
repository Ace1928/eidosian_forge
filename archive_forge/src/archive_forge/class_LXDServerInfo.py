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
class LXDServerInfo:
    """
    Wraps the response form /1.0
    """

    @classmethod
    def build_from_response(cls, metadata):
        server_info = LXDServerInfo()
        server_info.api_extensions = metadata.get('api_extensions', None)
        server_info.api_status = metadata.get('api_status', None)
        server_info.api_version = metadata.get('api_version', None)
        server_info.auth = metadata.get('auth', None)
        server_info.config = metadata.get('config', None)
        server_info.environment = metadata.get('environment', None)
        server_info.public = metadata.get('public', None)
        return server_info

    def __init__(self):
        self.api_extensions = None
        self.api_status = None
        self.api_version = None
        self.auth = None
        self.config = None
        self.environment = None
        self.public = None

    def __str__(self):
        return str(self.api_extensions) + str(self.api_status) + str(self.api_version) + str(self.auth) + str(self.config) + str(self.environment) + str(self.public)