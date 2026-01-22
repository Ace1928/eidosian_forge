from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def ListOverrides(self, name):
    """Calls the Security Profile Get API to list all Security Profile Overrides."""
    api_request = self.messages.NetworksecurityOrganizationsLocationsSecurityProfilesGetRequest(name=name)
    return self._security_profile_client.Get(api_request)