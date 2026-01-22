from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import sys
from googlecloudsdk.core import log
def GetSecurityPolicyId(org_security_policy_client, security_policy, organization=None):
    """Returns the security policy id that matches the display_name in the org.

  Args:
    org_security_policy_client: the organization security policy client.
    security_policy: the display name or ID of the security policy to be
      resolved.
    organization: the organization ID which the security policy belongs to.

  Returns:
    Security policy resource ID.
  """
    if not re.match('\\d{9,15}', security_policy):
        if organization is None:
            log.error('Must set --organization=ORGANIZATION when display name [%s]is used.', security_policy)
            sys.exit()
        return ResolveOrganizationSecurityPolicyId(org_security_policy_client, security_policy, ORGANIZATION_PREFIX + organization)
    return security_policy