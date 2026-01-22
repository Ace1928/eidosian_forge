from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2NodePool(_messages.Message):
    """Provides GKE node pool information.

  Fields:
    name: Kubernetes node pool name.
    nodes: Nodes associated with the finding.
  """
    name = _messages.StringField(1)
    nodes = _messages.MessageField('GoogleCloudSecuritycenterV2Node', 2, repeated=True)