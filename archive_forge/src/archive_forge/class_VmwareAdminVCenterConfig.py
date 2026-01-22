from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareAdminVCenterConfig(_messages.Message):
    """VmwareAdminVCenterConfig contains VCenter configuration for VMware admin
  cluster.

  Fields:
    address: The vCenter IP address.
    caCertData: Contains the vCenter CA certificate public key for SSL
      verification.
    cluster: The name of the vCenter cluster for the admin cluster.
    dataDisk: The name of the virtual machine disk (VMDK) for the admin
      cluster.
    datacenter: The name of the vCenter datacenter for the admin cluster.
    datastore: The name of the vCenter datastore for the admin cluster.
    folder: The name of the vCenter folder for the admin cluster.
    resourcePool: The name of the vCenter resource pool for the admin cluster.
    storagePolicyName: The name of the vCenter storage policy for the user
      cluster.
  """
    address = _messages.StringField(1)
    caCertData = _messages.StringField(2)
    cluster = _messages.StringField(3)
    dataDisk = _messages.StringField(4)
    datacenter = _messages.StringField(5)
    datastore = _messages.StringField(6)
    folder = _messages.StringField(7)
    resourcePool = _messages.StringField(8)
    storagePolicyName = _messages.StringField(9)