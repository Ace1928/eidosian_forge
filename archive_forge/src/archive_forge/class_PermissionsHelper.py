from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
class PermissionsHelper(object):
    """Get different kinds of permissions list from permissions provided.

  Attributes:
    messages: The iam messages.
    source_permissions: A list of permissions to inspect.
    testable_permissions_map: A dict maps from permissions name string to
        Permission message provided by the API.
  """

    def __init__(self, iam_client, messages, resource, permissions):
        """Create a PermissionsHelper object.

    To get the testable permissions for the given resource and store as a dict.

    Args:
      iam_client: The iam client.
      messages: The iam messages.
      resource: Resource reference for the project/organization whose
      permissions are being inspected.
      permissions: A list of permissions to inspect.
    """
        self.messages = messages
        self.source_permissions = permissions
        self.testable_permissions_map = {}
        if permissions:
            for permission in GetTestablePermissions(iam_client, messages, resource):
                self.testable_permissions_map[permission.name] = permission

    def GetTestingPermissions(self):
        """Returns the TESTING permissions among the permissions provided."""
        testing_permissions = []
        for permission in self.source_permissions:
            if permission in self.testable_permissions_map and self.testable_permissions_map[permission].customRolesSupportLevel == self.messages.Permission.CustomRolesSupportLevelValueValuesEnum.TESTING:
                testing_permissions.append(permission)
        return testing_permissions

    def GetValidPermissions(self):
        """Returns the valid permissions among the permissions provided."""
        valid_permissions = []
        for permission in self.source_permissions:
            if permission in self.testable_permissions_map and self.testable_permissions_map[permission].customRolesSupportLevel != self.messages.Permission.CustomRolesSupportLevelValueValuesEnum.NOT_SUPPORTED:
                valid_permissions.append(permission)
        return valid_permissions

    def GetNotSupportedPermissions(self):
        """Returns the not supported permissions among the permissions provided."""
        not_supported_permissions = []
        for permission in self.source_permissions:
            if permission in self.testable_permissions_map and self.testable_permissions_map[permission].customRolesSupportLevel == self.messages.Permission.CustomRolesSupportLevelValueValuesEnum.NOT_SUPPORTED:
                not_supported_permissions.append(permission)
        return not_supported_permissions

    def GetApiDisabledPermissons(self):
        """Returns the API disabled permissions among the permissions provided."""
        api_disabled_permissions = []
        for permission in self.source_permissions:
            if permission in self.testable_permissions_map and self.testable_permissions_map[permission].customRolesSupportLevel != self.messages.Permission.CustomRolesSupportLevelValueValuesEnum.NOT_SUPPORTED and self.testable_permissions_map[permission].apiDisabled:
                api_disabled_permissions.append(permission)
        return api_disabled_permissions

    def GetNotApplicablePermissions(self):
        """Returns the not applicable permissions among the permissions provided."""
        not_applicable_permissions = []
        for permission in self.source_permissions:
            if permission not in self.testable_permissions_map:
                not_applicable_permissions.append(permission)
        return not_applicable_permissions