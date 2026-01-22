from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import iam as kms_iam
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.command_lib.privateca import exceptions
def _CheckAllPermissions(actual_permissions, expected_permissions, resource):
    """Raises an exception if the expected permissions are not all present."""
    diff = set(expected_permissions) - set(actual_permissions)
    if diff:
        raise exceptions.InsufficientPermissionException(resource=resource, missing_permissions=diff)