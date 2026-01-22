from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1alpha2MarkRecommendationClaimedRequest(_messages.Message):
    """Request for the `MarkRecommendationClaimed` Method.

  Messages:
    StateMetadataValue: State properties to include with this state.
      Overwrites any existing `state_metadata`. Keys must match the regex
      `/^a-z0-9{0,62}$/`. Values must match the regex
      `/^[a-zA-Z0-9_./-]{0,255}$/`.

  Fields:
    etag: Fingerprint of the Recommendation. Provides optimistic locking.
    stateMetadata: State properties to include with this state. Overwrites any
      existing `state_metadata`. Keys must match the regex `/^a-z0-9{0,62}$/`.
      Values must match the regex `/^[a-zA-Z0-9_./-]{0,255}$/`.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class StateMetadataValue(_messages.Message):
        """State properties to include with this state. Overwrites any existing
    `state_metadata`. Keys must match the regex `/^a-z0-9{0,62}$/`. Values
    must match the regex `/^[a-zA-Z0-9_./-]{0,255}$/`.

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