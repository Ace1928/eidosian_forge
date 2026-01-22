from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
def _CheckInstancePermissions(self):
    """Check whether user has IAM permission on instance resource.

    Returns:
      set, missing IAM permissions.
    """
    response = self._ComputeTestIamPermissions(instance_permissions)
    return set(instance_permissions) - set(response.permissions)