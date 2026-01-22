from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
def DeletePosixAccounts(self, project_ref, operating_system=None):
    """Delete the posix accounts for an account in the current project.

    Args:
      project_ref: The oslogin.users.projects resource.
      operating_system: str, 'linux' or 'windows' (case insensitive).
    Returns:
      None
    """
    if operating_system:
        os_value = operating_system.upper()
        os_message = self.messages.OsloginUsersProjectsDeleteRequest.OperatingSystemTypeValueValuesEnum(os_value)
        message = self.messages.OsloginUsersProjectsDeleteRequest(name=project_ref.RelativeName(), operatingSystemType=os_message)
    else:
        message = self.messages.OsloginUsersProjectsDeleteRequest(name=project_ref.RelativeName())
    self.client.users_projects.Delete(message)