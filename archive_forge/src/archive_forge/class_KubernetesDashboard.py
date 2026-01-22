from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KubernetesDashboard(_messages.Message):
    """Configuration for the Kubernetes Dashboard.

  Fields:
    disabled: Whether the Kubernetes Dashboard is enabled for this cluster.
  """
    disabled = _messages.BooleanField(1)