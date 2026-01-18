import copy
import json
import email.utils
from typing import Dict, Optional
from libcloud.utils.py3 import httplib, urlquote
from libcloud.common.base import ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.common.google import GoogleAuthType, GoogleResponse, GoogleOAuth2Credential
from libcloud.storage.drivers.s3 import (

        Set the permissions for an ACL entity on a container or an object.

        :param container_name: The container name.
        :type container_name: ``str``

        :param object_name: The object name. Optional. Not providing an object
            will apply the acl to the container.
        :type object_name: ``str``

        :param entity: The entity to which apply the role. Optional. If not
            provided, the role will be applied to the authenticated user, if
            using an OAuth2 authentication scheme.
        :type entity: ``str``

        :param role: The permission/role to set on the entity.
        :type role: ``int`` from ContainerPermissions or ObjectPermissions
            or ``str``.

        :raises ValueError: If no entity was given, but was required. Or if
            the role isn't valid for the bucket or object.
        