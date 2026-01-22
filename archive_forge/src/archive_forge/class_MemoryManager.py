from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MemoryManager(_messages.Message):
    """The option enables the Kubernetes NUMA-aware Memory Manager feature.
  Detailed description about the feature can be found
  [here](https://kubernetes.io/docs/tasks/administer-cluster/memory-manager/).

  Fields:
    policy: Controls the memory management policy on the Node. See
      https://kubernetes.io/docs/tasks/administer-cluster/memory-
      manager/#policies The following values are allowed. * "none" * "static"
      The default value is 'none' if unspecified.
  """
    policy = _messages.StringField(1)