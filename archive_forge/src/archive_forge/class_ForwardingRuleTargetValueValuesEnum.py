from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ForwardingRuleTargetValueValuesEnum(_messages.Enum):
    """Output only. Specifies the type of the target of the forwarding rule.

    Values:
      FORWARDING_RULE_TARGET_UNSPECIFIED: Forwarding rule target is unknown.
      INSTANCE: Compute Engine instance for protocol forwarding.
      LOAD_BALANCER: Load Balancer. The specific type can be found from
        load_balancer_type.
      VPN_GATEWAY: Classic Cloud VPN Gateway.
      PSC: Forwarding Rule is a Private Service Connect endpoint.
    """
    FORWARDING_RULE_TARGET_UNSPECIFIED = 0
    INSTANCE = 1
    LOAD_BALANCER = 2
    VPN_GATEWAY = 3
    PSC = 4