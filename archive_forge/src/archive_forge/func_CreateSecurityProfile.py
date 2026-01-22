from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def CreateSecurityProfile(self, name, sp_id, parent, description, profile_type='THREAT_PREVENTION', labels=None):
    """Calls the Create Security Profile API."""
    security_profile = self.messages.SecurityProfile(name=name, description=description, type=self._ParseSecurityProfileType(profile_type), labels=labels)
    api_request = self.messages.NetworksecurityOrganizationsLocationsSecurityProfilesCreateRequest(parent=parent, securityProfile=security_profile, securityProfileId=sp_id)
    return self._security_profile_client.Create(api_request)