from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EffectiveQuotasForMetric(_messages.Message):
    """Effective quotas for a metric. It contains both the metadata for the
  metric as defined in the service config, and the effective limits for quota
  limits defined on the metric as calculated from service default, producer
  and consumer overrides, and adjusted by the reputation tier of the user.
  This is used only for quota limits that are grouped by metrics instead of
  quota groups.

  Messages:
    EffectiveLimitsValue: Effective limit values for all quota limits defined
      on the metric. The keys of the map are the name of the quota limits.

  Fields:
    effectiveLimits: Effective limit values for all quota limits defined on
      the metric. The keys of the map are the name of the quota limits.
    metric: The metric descriptor in service config.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EffectiveLimitsValue(_messages.Message):
        """Effective limit values for all quota limits defined on the metric. The
    keys of the map are the name of the quota limits.

    Messages:
      AdditionalProperty: An additional property for a EffectiveLimitsValue
        object.

    Fields:
      additionalProperties: Additional properties of type EffectiveLimitsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a EffectiveLimitsValue object.

      Fields:
        key: Name of the additional property.
        value: A EffectiveQuotaLimit2 attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('EffectiveQuotaLimit2', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    effectiveLimits = _messages.MessageField('EffectiveLimitsValue', 1)
    metric = _messages.MessageField('MetricDescriptor', 2)