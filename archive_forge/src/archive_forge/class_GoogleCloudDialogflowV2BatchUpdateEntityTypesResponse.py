from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2BatchUpdateEntityTypesResponse(_messages.Message):
    """The response message for EntityTypes.BatchUpdateEntityTypes.

  Fields:
    entityTypes: The collection of updated or created entity types.
  """
    entityTypes = _messages.MessageField('GoogleCloudDialogflowV2EntityType', 1, repeated=True)