from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RelationshipAttributes(_messages.Message):
    """DEPRECATED. This message only presents for the purpose of backward-
  compatibility. The server will never populate this message in responses. The
  relationship attributes which include `type`, `source_resource_type`,
  `target_resource_type` and `action`.

  Fields:
    action: The detail of the relationship, e.g. `contains`, `attaches`
    sourceResourceType: The source asset type. Example:
      `compute.googleapis.com/Instance`
    targetResourceType: The target asset type. Example:
      `compute.googleapis.com/Disk`
    type: The unique identifier of the relationship type. Example:
      `INSTANCE_TO_INSTANCEGROUP`
  """
    action = _messages.StringField(1)
    sourceResourceType = _messages.StringField(2)
    targetResourceType = _messages.StringField(3)
    type = _messages.StringField(4)