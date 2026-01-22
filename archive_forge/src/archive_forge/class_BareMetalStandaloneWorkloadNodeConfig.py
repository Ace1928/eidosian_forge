from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalStandaloneWorkloadNodeConfig(_messages.Message):
    """Specifies the workload node configurations.

  Fields:
    maxPodsPerNode: The maximum number of pods a node can run. The size of the
      CIDR range assigned to the node will be derived from this parameter.
  """
    maxPodsPerNode = _messages.IntegerField(1)