from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApplySoftwareUpdateRequest(_messages.Message):
    """Request for ApplySoftwareUpdate.

  Fields:
    applyAll: Whether to apply the update to all nodes. If set to true, will
      explicitly restrict users from specifying any nodes, and apply software
      update to all nodes (where applicable) within the instance.
    nodeIds: Nodes to which we should apply the update to. Note all the
      selected nodes are updated in parallel.
  """
    applyAll = _messages.BooleanField(1)
    nodeIds = _messages.StringField(2, repeated=True)