from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1ReplicaPlacement(_messages.Message):
    """Configuration for the placement of a control plane replica.

  Fields:
    azureAvailabilityZone: Required. For a given replica, the Azure
      availability zone where to provision the control plane VM and the ETCD
      disk.
    subnetId: Required. For a given replica, the ARM ID of the subnet where
      the control plane VM is deployed. Make sure it's a subnet under the
      virtual network in the cluster configuration.
  """
    azureAvailabilityZone = _messages.StringField(1)
    subnetId = _messages.StringField(2)