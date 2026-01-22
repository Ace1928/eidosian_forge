from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalAdminControlPlaneNodePoolConfig(_messages.Message):
    """BareMetalAdminControlPlaneNodePoolConfig specifies the control plane
  node pool configuration. We have a control plane specific node pool config
  so that we can flexible about supporting control plane specific fields in
  the future.

  Fields:
    nodePoolConfig: Required. The generic configuration for a node pool
      running the control plane.
  """
    nodePoolConfig = _messages.MessageField('BareMetalNodePoolConfig', 1)