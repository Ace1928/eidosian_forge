from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2BatchDeleteEntityTypesRequest(_messages.Message):
    """The request message for EntityTypes.BatchDeleteEntityTypes.

  Fields:
    entityTypeNames: Required. The names entity types to delete. All names
      must point to the same agent as `parent`.
  """
    entityTypeNames = _messages.StringField(1, repeated=True)