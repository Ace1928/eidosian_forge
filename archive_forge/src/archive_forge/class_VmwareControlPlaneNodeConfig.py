from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareControlPlaneNodeConfig(_messages.Message):
    """Specifies control plane node config for the VMware user cluster.

  Fields:
    autoResizeConfig: AutoResizeConfig provides auto resizing configurations.
    cpus: The number of CPUs for each admin cluster node that serve as control
      planes for this VMware user cluster. (default: 4 CPUs)
    memory: The megabytes of memory for each admin cluster node that serves as
      a control plane for this VMware user cluster (default: 8192 MB memory).
    replicas: The number of control plane nodes for this VMware user cluster.
      (default: 1 replica).
    vsphereConfig: Vsphere-specific config.
  """
    autoResizeConfig = _messages.MessageField('VmwareAutoResizeConfig', 1)
    cpus = _messages.IntegerField(2)
    memory = _messages.IntegerField(3)
    replicas = _messages.IntegerField(4)
    vsphereConfig = _messages.MessageField('VmwareControlPlaneVsphereConfig', 5)