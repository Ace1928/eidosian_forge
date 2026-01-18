from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import sys
from googlecloudsdk.core import log
Returns the security policy id that matches the display_name in the org.

  Args:
    org_security_policy_client: the organization security policy client.
    security_policy: the display name or ID of the security policy to be
      resolved.
    organization: the organization ID which the security policy belongs to.

  Returns:
    Security policy resource ID.
  