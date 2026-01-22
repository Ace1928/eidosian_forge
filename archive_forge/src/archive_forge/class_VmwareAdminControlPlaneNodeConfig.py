from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareAdminControlPlaneNodeConfig(_messages.Message):
    """VmwareAdminControlPlaneNodeConfig contains control plane node
  configuration for VMware admin cluster.

  Fields:
    cpus: The number of vCPUs for the control-plane node of the admin cluster.
    memory: The number of mebibytes of memory for the control-plane node of
      the admin cluster.
    replicas: The number of control plane nodes for this VMware admin cluster.
      (default: 1 replica).
  """
    cpus = _messages.IntegerField(1)
    memory = _messages.IntegerField(2)
    replicas = _messages.IntegerField(3)