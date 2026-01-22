from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartnerSSEGateway(_messages.Message):
    """Message describing PartnerSSEGateway object

  Messages:
    LabelsValue: Optional. Labels as key value pairs

  Fields:
    country: Output only. ISO-3166 alpha 2 country code used for localization.
      Filled from the customer SSEGateway, and only for PartnerSSEGateways
      associated with Symantec today.
    createTime: Output only. [Output only] Create time stamp
    labels: Optional. Labels as key value pairs
    maxBandwidthMbps: Output only. Not an enforced cap. Filled from the
      customer SSEGateway, and only for PartnerSSEGateways associated with
      Symantec today.
    name: Immutable. name of resource
    partnerSseEnvironment: Output only. [Output Only] Full URI of the partner
      environment this PartnerSSEGateway is connected to. Filled from the
      customer SSEGateway, and only for PartnerSSEGateways associated with
      Symantec today.
    partnerSseRealm: Output only. [Output Only] name of PartnerSSERealm owning
      the PartnerSSEGateway
    partnerSubnetRange: Optional. Subnet range of the partner-owned subnet.
    partnerVpcSubnetRange: Optional. Subnet range of the partner_vpc This
      field is deprecated. Use partner_subnet_range instead.
    sseBgpAsn: Output only. [Output Only] ASN of SSE BGP
    sseBgpIps: Output only. [Output Only] IP of SSE BGP
    sseGatewayReferenceId: Required. ID of the SSEGatewayReference that pairs
      with this PartnerSSEGateway
    sseNetwork: Output only. [Output Only] The ID of the network in
      sse_project containing sse_subnet_range. This is also known as the
      partnerFacingNetwork. Only filled for PartnerSSEGateways associated with
      Symantec today.
    sseProject: Output only. [Output Only] The project owning
      partner_facing_network. Only filled for PartnerSSEGateways associated
      with Symantec today.
    sseSubnetRange: Optional. Subnet range where SSE GW instances are
      deployed. Default value is set to "100.88.255.0/24". The CIDR suffix
      should be less than or equal to 24.
    sseTargetIp: Output only. [Output Only] Target IP that belongs to
      sse_subnet_range where partner should send the traffic to reach the
      customer networks.
    sseVpcSubnetRange: Output only. [Output Only] Subnet range of the subnet
      where partner traffic is routed. This field is deprecated. Use
      sse_subnet_range instead.
    sseVpcTargetIp: Output only. [Output Only] This is the IP where the
      partner traffic should be routed to. This field is deprecated. Use
      sse_target_ip instead.
    symantecOptions: Optional. Required iff Partner is Symantec.
    timezone: Output only. tzinfo identifier used for localization. Filled
      from the customer SSEGateway, and only for PartnerSSEGateways associated
      with Symantec today.
    updateTime: Output only. [Output only] Update time stamp
    vni: Optional. Virtual Network Identifier to use in NCG. Today the only
      partner that depends on it is Symantec.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels as key value pairs

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
    country = _messages.StringField(1)
    createTime = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    maxBandwidthMbps = _messages.IntegerField(4)
    name = _messages.StringField(5)
    partnerSseEnvironment = _messages.StringField(6)
    partnerSseRealm = _messages.StringField(7)
    partnerSubnetRange = _messages.StringField(8)
    partnerVpcSubnetRange = _messages.StringField(9)
    sseBgpAsn = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    sseBgpIps = _messages.StringField(11, repeated=True)
    sseGatewayReferenceId = _messages.StringField(12)
    sseNetwork = _messages.StringField(13)
    sseProject = _messages.StringField(14)
    sseSubnetRange = _messages.StringField(15)
    sseTargetIp = _messages.StringField(16)
    sseVpcSubnetRange = _messages.StringField(17)
    sseVpcTargetIp = _messages.StringField(18)
    symantecOptions = _messages.MessageField('PartnerSSEGatewayPartnerSSEGatewaySymantecOptions', 19)
    timezone = _messages.StringField(20)
    updateTime = _messages.StringField(21)
    vni = _messages.IntegerField(22, variant=_messages.Variant.INT32)