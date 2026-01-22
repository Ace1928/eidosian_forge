from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def GetSecurityProfileGroup(self, security_profile_group_name):
    """Calls the Security Profile Group Get API.

    Args:
      security_profile_group_name: Fully specified Security Profile Group.

    Returns:
      Security Profile Group object.
    """
    api_request = self.messages.NetworksecurityOrganizationsLocationsSecurityProfileGroupsGetRequest(name=security_profile_group_name)
    return self._security_profile_group_client.Get(api_request)