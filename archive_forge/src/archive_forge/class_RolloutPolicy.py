from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RolloutPolicy(_messages.Message):
    """A rollout policy configuration.

  Messages:
    LocationRolloutPoliciesValue: Location based rollout policies to apply to
      the resource. Currently only zone names are supported and must be
      represented as valid URLs, like: zones/us-central1-a. The value expects
      an RFC3339 timestamp on or after which the update is considered rolled
      out to the specified location.

  Fields:
    defaultRolloutTime: An optional RFC3339 timestamp on or after which the
      update is considered rolled out to any zone that is not explicitly
      stated.
    locationRolloutPolicies: Location based rollout policies to apply to the
      resource. Currently only zone names are supported and must be
      represented as valid URLs, like: zones/us-central1-a. The value expects
      an RFC3339 timestamp on or after which the update is considered rolled
      out to the specified location.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LocationRolloutPoliciesValue(_messages.Message):
        """Location based rollout policies to apply to the resource. Currently
    only zone names are supported and must be represented as valid URLs, like:
    zones/us-central1-a. The value expects an RFC3339 timestamp on or after
    which the update is considered rolled out to the specified location.

    Messages:
      AdditionalProperty: An additional property for a
        LocationRolloutPoliciesValue object.

    Fields:
      additionalProperties: Additional properties of type
        LocationRolloutPoliciesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LocationRolloutPoliciesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    defaultRolloutTime = _messages.StringField(1)
    locationRolloutPolicies = _messages.MessageField('LocationRolloutPoliciesValue', 2)