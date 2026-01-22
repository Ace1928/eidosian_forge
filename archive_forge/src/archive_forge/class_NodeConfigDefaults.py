from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeConfigDefaults(_messages.Message):
    """Subset of NodeConfig message that has defaults.

  Fields:
    containerdConfig: Parameters for containerd customization.
    gcfsConfig: GCFS (Google Container File System, also known as Riptide)
      options.
    loggingConfig: Logging configuration for node pools.
    nodeKubeletConfig: NodeKubeletConfig controls the defaults for new node-
      pools. Currently only `insecure_kubelet_readonly_port_enabled` can be
      set here.
  """
    containerdConfig = _messages.MessageField('ContainerdConfig', 1)
    gcfsConfig = _messages.MessageField('GcfsConfig', 2)
    loggingConfig = _messages.MessageField('NodePoolLoggingConfig', 3)
    nodeKubeletConfig = _messages.MessageField('NodeKubeletConfig', 4)