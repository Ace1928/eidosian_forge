from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1DeleteFeatureValuesRequestSelectEntity(_messages.Message):
    """Message to select entity. If an entity id is selected, all the feature
  values corresponding to the entity id will be deleted, including the
  entityId.

  Fields:
    entityIdSelector: Required. Selectors choosing feature values of which
      entity id to be deleted from the EntityType.
  """
    entityIdSelector = _messages.MessageField('GoogleCloudAiplatformV1beta1EntityIdSelector', 1)