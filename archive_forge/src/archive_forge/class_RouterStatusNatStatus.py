from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouterStatusNatStatus(_messages.Message):
    """Status of a NAT contained in this router.

  Fields:
    autoAllocatedNatIps: A list of IPs auto-allocated for NAT. Example:
      ["1.1.1.1", "129.2.16.89"]
    drainAutoAllocatedNatIps: A list of IPs auto-allocated for NAT that are in
      drain mode. Example: ["1.1.1.1", "179.12.26.133"].
    drainUserAllocatedNatIps: A list of IPs user-allocated for NAT that are in
      drain mode. Example: ["1.1.1.1", "179.12.26.133"].
    minExtraNatIpsNeeded: The number of extra IPs to allocate. This will be
      greater than 0 only if user-specified IPs are NOT enough to allow all
      configured VMs to use NAT. This value is meaningful only when auto-
      allocation of NAT IPs is *not* used.
    name: Unique name of this NAT.
    numVmEndpointsWithNatMappings: Number of VM endpoints (i.e., Nics) that
      can use NAT.
    ruleStatus: Status of rules in this NAT.
    userAllocatedNatIpResources: A list of fully qualified URLs of reserved IP
      address resources.
    userAllocatedNatIps: A list of IPs user-allocated for NAT. They will be
      raw IP strings like "179.12.26.133".
  """
    autoAllocatedNatIps = _messages.StringField(1, repeated=True)
    drainAutoAllocatedNatIps = _messages.StringField(2, repeated=True)
    drainUserAllocatedNatIps = _messages.StringField(3, repeated=True)
    minExtraNatIpsNeeded = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    name = _messages.StringField(5)
    numVmEndpointsWithNatMappings = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    ruleStatus = _messages.MessageField('RouterStatusNatStatusNatRuleStatus', 7, repeated=True)
    userAllocatedNatIpResources = _messages.StringField(8, repeated=True)
    userAllocatedNatIps = _messages.StringField(9, repeated=True)