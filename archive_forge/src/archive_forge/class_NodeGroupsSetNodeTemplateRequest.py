from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeGroupsSetNodeTemplateRequest(_messages.Message):
    """A NodeGroupsSetNodeTemplateRequest object.

  Fields:
    nodeTemplate: Full or partial URL of the node template resource to be
      updated for this node group.
  """
    nodeTemplate = _messages.StringField(1)