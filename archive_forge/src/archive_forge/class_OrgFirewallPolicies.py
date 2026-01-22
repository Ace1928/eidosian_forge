from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class OrgFirewallPolicies(base.Group):
    """Manage Compute Engine organization firewall policies.

  Manage Compute Engine organization firewall policies. Organization
  firewall policies are used to control incoming/outgoing traffic.
  """
    category = base.COMPUTE_CATEGORY