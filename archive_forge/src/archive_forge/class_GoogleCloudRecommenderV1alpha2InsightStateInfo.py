from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1alpha2InsightStateInfo(_messages.Message):
    """Information related to insight state.

  Enums:
    StateValueValuesEnum: Insight state.

  Messages:
    StateMetadataValue: A map of metadata for the state, provided by user or
      automations systems.

  Fields:
    state: Insight state.
    stateMetadata: A map of metadata for the state, provided by user or
      automations systems.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Insight state.

    Values:
      STATE_UNSPECIFIED: Unspecified state.
      ACTIVE: Insight is active. Content for ACTIVE insights can be updated by
        Google. ACTIVE insights can be marked DISMISSED OR ACCEPTED.
      ACCEPTED: Some action has been taken based on this insight. Insights
        become accepted when a recommendation derived from the insight has
        been marked CLAIMED, SUCCEEDED, or FAILED. ACTIVE insights can also be
        marked ACCEPTED explicitly. Content for ACCEPTED insights is
        immutable. ACCEPTED insights can only be marked ACCEPTED (which may
        update state metadata).
      DISMISSED: Insight is dismissed. Content for DISMISSED insights can be
        updated by Google. DISMISSED insights can be marked as ACTIVE.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        ACCEPTED = 2
        DISMISSED = 3

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