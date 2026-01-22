from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
def _ResourceManagerTestIamPermissions(self, permissions):
    """Check whether user has IAM permission on resource manager.

    Args:
      permissions: list, the permissions to check for the project resource.

    Returns:
      set, missing IAM permissions.
    """
    iam_request = self.resourcemanager_message_v3.TestIamPermissionsRequest(permissions=permissions)
    request = self.resourcemanager_message_v3.CloudresourcemanagerProjectsTestIamPermissionsRequest(resource='projects/{project}'.format(project=self.project.name), testIamPermissionsRequest=iam_request)
    return self.resourcemanager_client_v3.projects.TestIamPermissions(request)