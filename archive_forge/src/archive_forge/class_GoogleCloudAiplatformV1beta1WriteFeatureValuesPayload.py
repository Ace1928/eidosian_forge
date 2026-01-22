from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1WriteFeatureValuesPayload(_messages.Message):
    """Contains Feature values to be written for a specific entity.

  Messages:
    FeatureValuesValue: Required. Feature values to be written, mapping from
      Feature ID to value. Up to 100,000 `feature_values` entries may be
      written across all payloads. The feature generation time, aligned by
      days, must be no older than five years (1825 days) and no later than one
      year (366 days) in the future.

  Fields:
    entityId: Required. The ID of the entity.
    featureValues: Required. Feature values to be written, mapping from
      Feature ID to value. Up to 100,000 `feature_values` entries may be
      written across all payloads. The feature generation time, aligned by
      days, must be no older than five years (1825 days) and no later than one
      year (366 days) in the future.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class FeatureValuesValue(_messages.Message):
        """Required. Feature values to be written, mapping from Feature ID to
    value. Up to 100,000 `feature_values` entries may be written across all
    payloads. The feature generation time, aligned by days, must be no older
    than five years (1825 days) and no later than one year (366 days) in the
    future.

    Messages:
      AdditionalProperty: An additional property for a FeatureValuesValue
        object.

    Fields:
      additionalProperties: Additional properties of type FeatureValuesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a FeatureValuesValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudAiplatformV1beta1FeatureValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudAiplatformV1beta1FeatureValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    entityId = _messages.StringField(1)
    featureValues = _messages.MessageField('FeatureValuesValue', 2)