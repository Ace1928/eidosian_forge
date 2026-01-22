from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HierarchyControllerConfig(_messages.Message):
    """Configuration for Hierarchy Controller

  Fields:
    enableHierarchicalResourceQuota: Whether hierarchical resource quota is
      enabled in this cluster.
    enablePodTreeLabels: Whether pod tree labels are enabled in this cluster.
    enabled: Whether Hierarchy Controller is enabled in this cluster.
  """
    enableHierarchicalResourceQuota = _messages.BooleanField(1)
    enablePodTreeLabels = _messages.BooleanField(2)
    enabled = _messages.BooleanField(3)