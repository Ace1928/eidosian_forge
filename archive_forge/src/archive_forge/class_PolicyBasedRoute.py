from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyBasedRoute(_messages.Message):
    """Policy-based routes route L4 network traffic based on not just
  destination IP address, but also source IP address, protocol, and more. If a
  policy-based route conflicts with other types of routes, the policy-based
  route always take precedence.

  Enums:
    NextHopOtherRoutesValueValuesEnum: Optional. Other routes that will be
      referenced to determine the next hop of the packet.

  Messages:
    LabelsValue: User-defined labels.

  Fields:
    createTime: Output only. Time when the policy-based route was created.
    description: Optional. An optional description of this resource. Provide
      this field when you create the resource.
    filter: Required. The filter to match L4 traffic.
    interconnectAttachment: Optional. The interconnect attachments that this
      policy-based route applies to.
    kind: Output only. Type of this resource. Always
      networkconnectivity#policyBasedRoute for policy-based Route resources.
    labels: User-defined labels.
    name: Immutable. A unique name of the resource in the form of `projects/{p
      roject_number}/locations/global/PolicyBasedRoutes/{policy_based_route_id
      }`
    network: Required. Fully-qualified URL of the network that this route
      applies to, for example: projects/my-project/global/networks/my-network.
    nextHopIlbIp: Optional. The IP address of a global-access-enabled L4 ILB
      that is the next hop for matching packets. For this version, only
      nextHopIlbIp is supported.
    nextHopOtherRoutes: Optional. Other routes that will be referenced to
      determine the next hop of the packet.
    priority: Optional. The priority of this policy-based route. Priority is
      used to break ties in cases where there are more than one matching
      policy-based routes found. In cases where multiple policy-based routes
      are matched, the one with the lowest-numbered priority value wins. The
      default value is 1000. The priority value must be from 1 to 65535,
      inclusive.
    selfLink: Output only. Server-defined fully-qualified URL for this
      resource.
    updateTime: Output only. Time when the policy-based route was updated.
    virtualMachine: Optional. VM instances to which this policy-based route
      applies to.
    warnings: Output only. If potential misconfigurations are detected for
      this route, this field will be populated with warning messages.
  """

    class NextHopOtherRoutesValueValuesEnum(_messages.Enum):
        """Optional. Other routes that will be referenced to determine the next
    hop of the packet.

    Values:
      OTHER_ROUTES_UNSPECIFIED: Default value.
      DEFAULT_ROUTING: Use the routes from the default routing tables (system-
        generated routes, custom routes, peering route) to determine the next
        hop. This will effectively exclude matching packets being applied on
        other PBRs with a lower priority.
    """
        OTHER_ROUTES_UNSPECIFIED = 0
        DEFAULT_ROUTING = 1

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """User-defined labels.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    filter = _messages.MessageField('Filter', 3)
    interconnectAttachment = _messages.MessageField('InterconnectAttachment', 4)
    kind = _messages.StringField(5)
    labels = _messages.MessageField('LabelsValue', 6)
    name = _messages.StringField(7)
    network = _messages.StringField(8)
    nextHopIlbIp = _messages.StringField(9)
    nextHopOtherRoutes = _messages.EnumField('NextHopOtherRoutesValueValuesEnum', 10)
    priority = _messages.IntegerField(11, variant=_messages.Variant.INT32)
    selfLink = _messages.StringField(12)
    updateTime = _messages.StringField(13)
    virtualMachine = _messages.MessageField('VirtualMachine', 14)
    warnings = _messages.MessageField('Warnings', 15, repeated=True)