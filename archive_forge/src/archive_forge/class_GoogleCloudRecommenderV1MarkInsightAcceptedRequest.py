from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1MarkInsightAcceptedRequest(_messages.Message):
    """Request for the `MarkInsightAccepted` method.

  Messages:
    StateMetadataValue: Optional. State properties user wish to include with
      this state. Full replace of the current state_metadata.

  Fields:
    etag: Required. Fingerprint of the Insight. Provides optimistic locking.
    stateMetadata: Optional. State properties user wish to include with this
      state. Full replace of the current state_metadata.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class StateMetadataValue(_messages.Message):
        """Optional. State properties user wish to include with this state. Full
    replace of the current state_metadata.

    Messages:
      AdditionalProperty: An additional property for a StateMetadataValue
        object.

    Fields:
      additionalProperties: Additional properties of type StateMetadataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a StateMetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    etag = _messages.StringField(1)
    stateMetadata = _messages.MessageField('StateMetadataValue', 2)