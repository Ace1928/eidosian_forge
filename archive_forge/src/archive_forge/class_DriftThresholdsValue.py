from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class DriftThresholdsValue(_messages.Message):
    """Key is the feature name and value is the threshold. If a feature needs
    to be monitored for drift, a value threshold must be configured for that
    feature. The threshold here is against feature distribution distance
    between different time windws.

    Messages:
      AdditionalProperty: An additional property for a DriftThresholdsValue
        object.

    Fields:
      additionalProperties: Additional properties of type DriftThresholdsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a DriftThresholdsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudAiplatformV1ThresholdConfig attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleCloudAiplatformV1ThresholdConfig', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)