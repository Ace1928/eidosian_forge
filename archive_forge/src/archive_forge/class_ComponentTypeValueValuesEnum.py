from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComponentTypeValueValuesEnum(_messages.Enum):
    """Output only. Type of component.

    Values:
      VMWARE_COMPONENT_TYPE_UNSPECIFIED: The default value. This value should
        never be used.
      VCENTER: Vcenter server.
      ESXI: Esxi nodes + Transport nodes upgrade.
      NSXT_UC: Nsxt upgrade coordinator.
      NSXT_EDGE: Nsxt edges cluster.
      NSXT_MGR: Nsxt managers/management plane.
      HCX: HCX appliance.
      VSAN: VSAN cluster.
      DVS: DVS switch.
      NAMESERVER_VM: Nameserver VMs.
      KMS_VM: KMS VM used for vsan encryption.
      WITNESS_VM: Witness VM in case of stretch PC.
      NSXT: nsxt
    """
    VMWARE_COMPONENT_TYPE_UNSPECIFIED = 0
    VCENTER = 1
    ESXI = 2
    NSXT_UC = 3
    NSXT_EDGE = 4
    NSXT_MGR = 5
    HCX = 6
    VSAN = 7
    DVS = 8
    NAMESERVER_VM = 9
    KMS_VM = 10
    WITNESS_VM = 11
    NSXT = 12