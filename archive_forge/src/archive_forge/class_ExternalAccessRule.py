from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExternalAccessRule(_messages.Message):
    """External access firewall rules for filtering incoming traffic destined
  to `ExternalAddress` resources.

  Enums:
    ActionValueValuesEnum: The action that the external access rule performs.
    StateValueValuesEnum: Output only. The state of the resource.

  Fields:
    action: The action that the external access rule performs.
    createTime: Output only. Creation time of this resource.
    description: User-provided description for this external access rule.
    destinationIpRanges: If destination ranges are specified, the external
      access rule applies only to the traffic that has a destination IP
      address in these ranges. The specified IP addresses must have reserved
      external IP addresses in the scope of the parent network policy. To
      match all external IP addresses in the scope of the parent network
      policy, specify `0.0.0.0/0`. To match a specific external IP address,
      specify it using the `IpRange.external_address` property.
    destinationPorts: A list of destination ports to which the external access
      rule applies. This field is only applicable for the UDP or TCP protocol.
      Each entry must be either an integer or a range. For example: `["22"]`,
      `["80","443"]`, or `["12345-12349"]`. To match all destination ports,
      specify `["0-65535"]`.
    ipProtocol: The IP protocol to which the external access rule applies.
      This value can be one of the following three protocol strings (not case-
      sensitive): `tcp`, `udp`, or `icmp`.
    name: Output only. The resource name of this external access rule.
      Resource names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1/networkPolicies/my-
      policy/externalAccessRules/my-rule`
    priority: External access rule priority, which determines the external
      access rule to use when multiple rules apply. If multiple rules have the
      same priority, their ordering is non-deterministic. If specific ordering
      is required, assign unique priorities to enforce such ordering. The
      external access rule priority is an integer from 100 to 4096, both
      inclusive. Lower integers indicate higher precedence. For example, a
      rule with priority `100` has higher precedence than a rule with priority
      `101`.
    sourceIpRanges: If source ranges are specified, the external access rule
      applies only to traffic that has a source IP address in these ranges.
      These ranges can either be expressed in the CIDR format or as an IP
      address. As only inbound rules are supported, `ExternalAddress`
      resources cannot be the source IP addresses of an external access rule.
      To match all source addresses, specify `0.0.0.0/0`.
    sourcePorts: A list of source ports to which the external access rule
      applies. This field is only applicable for the UDP or TCP protocol. Each
      entry must be either an integer or a range. For example: `["22"]`,
      `["80","443"]`, or `["12345-12349"]`. To match all source ports, specify
      `["0-65535"]`.
    state: Output only. The state of the resource.
    uid: Output only. System-generated unique identifier for the resource.
    updateTime: Output only. Last update time of this resource.
  """

    class ActionValueValuesEnum(_messages.Enum):
        """The action that the external access rule performs.

    Values:
      ACTION_UNSPECIFIED: Defaults to allow.
      ALLOW: Allows connections that match the other specified components.
      DENY: Blocks connections that match the other specified components.
    """
        ACTION_UNSPECIFIED = 0
        ALLOW = 1
        DENY = 2

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the resource.

    Values:
      STATE_UNSPECIFIED: The default value. This value is used if the state is
        omitted.
      ACTIVE: The rule is ready.
      CREATING: The rule is being created.
      UPDATING: The rule is being updated.
      DELETING: The rule is being deleted.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        CREATING = 2
        UPDATING = 3
        DELETING = 4
    action = _messages.EnumField('ActionValueValuesEnum', 1)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    destinationIpRanges = _messages.MessageField('IpRange', 4, repeated=True)
    destinationPorts = _messages.StringField(5, repeated=True)
    ipProtocol = _messages.StringField(6)
    name = _messages.StringField(7)
    priority = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    sourceIpRanges = _messages.MessageField('IpRange', 9, repeated=True)
    sourcePorts = _messages.StringField(10, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 11)
    uid = _messages.StringField(12)
    updateTime = _messages.StringField(13)