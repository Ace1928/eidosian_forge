from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def UpdateSecurityProfileGroup(self, security_profile_group_name, description, threat_prevention_profile, update_mask, labels=None):
    """Calls the Patch Security Profile Group API."""
    security_profile_group = self.messages.SecurityProfileGroup(name=security_profile_group_name, description=description, threatPreventionProfile=threat_prevention_profile, labels=labels)
    api_request = self.messages.NetworksecurityOrganizationsLocationsSecurityProfileGroupsPatchRequest(name=security_profile_group_name, securityProfileGroup=security_profile_group, updateMask=update_mask)
    return self._security_profile_group_client.Patch(api_request)