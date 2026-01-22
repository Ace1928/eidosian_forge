from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodePoolAutoConfig(_messages.Message):
    """Node pool configs that apply to all auto-provisioned node pools in
  autopilot clusters and node auto-provisioning enabled clusters.

  Fields:
    networkTags: The list of instance tags applied to all nodes. Tags are used
      to identify valid sources or targets for network firewalls and are
      specified by the client during cluster creation. Each tag within the
      list must comply with RFC1035.
    nodeKubeletConfig: NodeKubeletConfig controls the defaults for
      autoprovisioned node-pools. Currently only
      `insecure_kubelet_readonly_port_enabled` can be set here.
    resourceManagerTags: Resource manager tag keys and values to be attached
      to the nodes for managing Compute Engine firewalls using Network
      Firewall Policies.
  """
    networkTags = _messages.MessageField('NetworkTags', 1)
    nodeKubeletConfig = _messages.MessageField('NodeKubeletConfig', 2)
    resourceManagerTags = _messages.MessageField('ResourceManagerTags', 3)