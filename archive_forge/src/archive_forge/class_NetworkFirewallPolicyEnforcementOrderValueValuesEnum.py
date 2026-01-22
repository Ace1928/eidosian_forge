from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkFirewallPolicyEnforcementOrderValueValuesEnum(_messages.Enum):
    """The network firewall policy enforcement order. Can be either
    AFTER_CLASSIC_FIREWALL or BEFORE_CLASSIC_FIREWALL. Defaults to
    AFTER_CLASSIC_FIREWALL if the field is not specified.

    Values:
      AFTER_CLASSIC_FIREWALL: <no description>
      BEFORE_CLASSIC_FIREWALL: <no description>
    """
    AFTER_CLASSIC_FIREWALL = 0
    BEFORE_CLASSIC_FIREWALL = 1