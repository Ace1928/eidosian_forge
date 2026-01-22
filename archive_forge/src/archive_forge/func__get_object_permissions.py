import copy
import json
import email.utils
from typing import Dict, Optional
from libcloud.utils.py3 import httplib, urlquote
from libcloud.common.base import ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.common.google import GoogleAuthType, GoogleResponse, GoogleOAuth2Credential
from libcloud.storage.drivers.s3 import (
def _get_object_permissions(self, container_name, object_name):
    """
        Return the object permissions for the current authenticated user.
        If the object does not exist, or no object_name is given, return the
        default object permissions.

        :param container_name: The container name.
        :type container_name: ``str``

        :param object_name: The object name.
        :type object_name: ``str``

        :return: The permissions on the object or default object permissions.
        :rtype: ``int`` from ObjectPermissions
        """
    try:
        self.json_connection.request('/storage/v1/b/{}/o/{}/acl'.format(container_name, object_name))
        return ObjectPermissions.OWNER
    except ProviderError as e:
        if e.http_code not in [httplib.FORBIDDEN, httplib.NOT_FOUND]:
            raise
    try:
        self.json_connection.request('/storage/v1/b/{}/o/{}'.format(container_name, object_name))
        return ObjectPermissions.READER
    except ProviderError as e:
        if e.http_code not in [httplib.FORBIDDEN, httplib.NOT_FOUND]:
            raise
    return ObjectPermissions.NONE