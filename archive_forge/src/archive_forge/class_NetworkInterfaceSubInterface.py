from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkInterfaceSubInterface(_messages.Message):
    """A NetworkInterfaceSubInterface object.

  Enums:
    IpAllocationModeValueValuesEnum:

  Fields:
    ipAddress: An IPv4 internal IP address to assign to the instance for this
      subinterface. If specified, ip_allocation_mode should be set to
      ALLOCATE_IP.
    ipAllocationMode: A IpAllocationModeValueValuesEnum attribute.
    subnetwork: If specified, this subnetwork must belong to the same network
      as that of the network interface. If not specified the subnet of network
      interface will be used. If you specify this property, you can specify
      the subnetwork as a full or partial URL. For example, the following are
      all valid URLs: -
      https://www.googleapis.com/compute/v1/projects/project/regions/region
      /subnetworks/subnetwork - regions/region/subnetworks/subnetwork
    vlan: VLAN tag. Should match the VLAN(s) supported by the subnetwork to
      which this subinterface is connecting.
  """

    class IpAllocationModeValueValuesEnum(_messages.Enum):
        """IpAllocationModeValueValuesEnum enum type.

    Values:
      ALLOCATE_IP: Allocates an internal IPv4 IP address from subnets
        secondary IP Range.
      DO_NOT_ALLOCATE_IP: No IP allocation is done for the subinterface.
      UNSPECIFIED: <no description>
    """
        ALLOCATE_IP = 0
        DO_NOT_ALLOCATE_IP = 1
        UNSPECIFIED = 2
    ipAddress = _messages.StringField(1)
    ipAllocationMode = _messages.EnumField('IpAllocationModeValueValuesEnum', 2)
    subnetwork = _messages.StringField(3)
    vlan = _messages.IntegerField(4, variant=_messages.Variant.INT32)