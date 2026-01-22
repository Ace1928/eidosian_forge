from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirewallRuleTypeValueValuesEnum(_messages.Enum):
    """The firewall rule's type.

    Values:
      FIREWALL_RULE_TYPE_UNSPECIFIED: Unspecified type.
      HIERARCHICAL_FIREWALL_POLICY_RULE: Hierarchical firewall policy rule.
        For details, see [Hierarchical firewall policies
        overview](https://cloud.google.com/vpc/docs/firewall-policies).
      VPC_FIREWALL_RULE: VPC firewall rule. For details, see [VPC firewall
        rules overview](https://cloud.google.com/vpc/docs/firewalls).
      IMPLIED_VPC_FIREWALL_RULE: Implied VPC firewall rule. For details, see
        [Implied rules](https://cloud.google.com/vpc/docs/firewalls#default_fi
        rewall_rules).
      SERVERLESS_VPC_ACCESS_MANAGED_FIREWALL_RULE: Implicit firewall rules
        that are managed by serverless VPC access to allow ingress access.
        They are not visible in the Google Cloud console. For details, see
        [VPC connector's implicit
        rules](https://cloud.google.com/functions/docs/networking/connecting-
        vpc#restrict-access).
      NETWORK_FIREWALL_POLICY_RULE: Global network firewall policy rule. For
        details, see [Network firewall
        policies](https://cloud.google.com/vpc/docs/network-firewall-
        policies).
      NETWORK_REGIONAL_FIREWALL_POLICY_RULE: Regional network firewall policy
        rule. For details, see [Regional network firewall
        policies](https://cloud.google.com/firewall/docs/regional-firewall-
        policies).
      UNSUPPORTED_FIREWALL_POLICY_RULE: Firewall policy rule containing
        attributes not yet supported in Connectivity tests. Firewall analysis
        is skipped if such a rule can potentially be matched. Please see the
        [list of unsupported configurations](https://cloud.google.com/network-
        intelligence-center/docs/connectivity-
        tests/concepts/overview#unsupported-configs).
      TRACKING_STATE: Tracking state for response traffic created when request
        traffic goes through allow firewall rule. For details, see [firewall
        rules specifications](https://cloud.google.com/firewall/docs/firewalls
        #specifications)
    """
    FIREWALL_RULE_TYPE_UNSPECIFIED = 0
    HIERARCHICAL_FIREWALL_POLICY_RULE = 1
    VPC_FIREWALL_RULE = 2
    IMPLIED_VPC_FIREWALL_RULE = 3
    SERVERLESS_VPC_ACCESS_MANAGED_FIREWALL_RULE = 4
    NETWORK_FIREWALL_POLICY_RULE = 5
    NETWORK_REGIONAL_FIREWALL_POLICY_RULE = 6
    UNSUPPORTED_FIREWALL_POLICY_RULE = 7
    TRACKING_STATE = 8