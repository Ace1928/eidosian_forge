from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import sys
from googlecloudsdk.api_lib import network_security
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import reference_utils
from googlecloudsdk.core import log
def BuildSecurityProfileGroupUrl(security_profile_group, optional_organization, firewall_policy_client, firewall_policy_id):
    """Returns Full URL reference of Security Profile Group.

  Args:
    security_profile_group: reference string provided by the user.
    optional_organization: organization if provided.
    firewall_policy_client: the organization firewall policy client.
    firewall_policy_id: the short name or ID of the firewall policy to be
      resolved.

  Returns:
    URL to Security Profile Group.
  """
    if '/' in security_profile_group:
        return security_profile_group
    organization = GetFirewallPolicyOrganization(firewall_policy_client=firewall_policy_client, firewall_policy_id=firewall_policy_id, optional_organization=optional_organization)
    return reference_utils.BuildFullResourceUrlForOrgBasedResource(base_uri=network_security.GetApiBaseUrl(network_security.base.ReleaseTrack.GA), org_id=organization, collection_name='securityProfileGroups', resource_name=security_profile_group)