from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1alpha2RecommendationStateInfo(_messages.Message):
    """Information for state. Contains state and metadata.

  Enums:
    StateValueValuesEnum: The state of the recommendation, Eg ACTIVE,
      SUCCEEDED, FAILED.

  Messages:
    StateMetadataValue: A map of metadata for the state, provided by user or
      automations systems.

  Fields:
    state: The state of the recommendation, Eg ACTIVE, SUCCEEDED, FAILED.
    stateMetadata: A map of metadata for the state, provided by user or
      automations systems.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The state of the recommendation, Eg ACTIVE, SUCCEEDED, FAILED.

    Values:
      STATE_UNSPECIFIED: Default state. Don't use directly.
      ACTIVE: Recommendation is active and can be applied. Recommendations
        content can be updated by Google. ACTIVE recommendations can be marked
        as CLAIMED, SUCCEEDED, or FAILED.
      CLAIMED: Recommendation is in claimed state. Recommendations content is
        immutable and cannot be updated by Google. CLAIMED recommendations can
        be marked as CLAIMED, SUCCEEDED, or FAILED.
      SUCCEEDED: Recommendation is in succeeded state. Recommendations content
        is immutable and cannot be updated by Google. SUCCEEDED
        recommendations can be marked as SUCCEEDED, or FAILED.
      FAILED: Recommendation is in failed state. Recommendations content is
        immutable and cannot be updated by Google. FAILED recommendations can
        be marked as SUCCEEDED, or FAILED.
      DISMISSED: Recommendation is in dismissed state. Recommendation content
        can be updated by Google. DISMISSED recommendations can be marked as
        ACTIVE.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        CLAIMED = 2
        SUCCEEDED = 3
        FAILED = 4
        DISMISSED = 5

    @encoding.MapUnrecognizedFields('additionalProperties')
    class StateMetadataValue(_messages.Message):
        """A map of metadata for the state, provided by user or automations
    systems.

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
    state = _messages.EnumField('StateValueValuesEnum', 1)
    stateMetadata = _messages.MessageField('StateMetadataValue', 2)