from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
def _CheckOsLoginPermissions(self):
    """Check whether user has oslogin IAM permissions.

    Returns:
      set, missing IAM permissions.
    """
    response = self._ComputeTestIamPermissions(oslogin_permissions)
    return set(oslogin_permissions) - set(response.permissions)