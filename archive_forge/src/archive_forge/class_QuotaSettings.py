from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QuotaSettings(_messages.Message):
    """Per-consumer overrides for quota settings. See google/api/quota.proto
  for the corresponding service configuration which provides the default
  values.

  Messages:
    ConsumerOverridesValue: Quota overrides set by the consumer. Consumer
      overrides will only have an effect up to the max_limit specified in the
      service config, or the the producer override, if one exists.  The key
      for this map is one of the following:  - '<GROUP_NAME>/<LIMIT_NAME>' for
      quotas defined within quota groups, where GROUP_NAME is the
      google.api.QuotaGroup.name field and LIMIT_NAME is the
      google.api.QuotaLimit.name field from the service config.  For example:
      'ReadGroup/ProjectDaily'.  - '<LIMIT_NAME>' for quotas defined without
      quota groups, where LIMIT_NAME is the google.api.QuotaLimit.name field
      from the service config. For example: 'borrowedCountPerOrganization'.
    EffectiveQuotasValue: The effective quota limits for each group, derived
      from the service defaults together with any producer or consumer
      overrides. For each limit, the effective value is the minimum of the
      producer and consumer overrides if either is present, or else the
      service default if neither is present. DEPRECATED. Use
      effective_quota_groups instead.
    ProducerOverridesValue: Quota overrides set by the producer. Note that if
      a consumer override is also specified, then the minimum of the two will
      be used. This allows consumers to cap their usage voluntarily.  The key
      for this map is one of the following:  - '<GROUP_NAME>/<LIMIT_NAME>' for
      quotas defined within quota groups, where GROUP_NAME is the
      google.api.QuotaGroup.name field and LIMIT_NAME is the
      google.api.QuotaLimit.name field from the service config.  For example:
      'ReadGroup/ProjectDaily'.  - '<LIMIT_NAME>' for quotas defined without
      quota groups, where LIMIT_NAME is the google.api.QuotaLimit.name field
      from the service config. For example: 'borrowedCountPerOrganization'.

  Fields:
    consumerOverrides: Quota overrides set by the consumer. Consumer overrides
      will only have an effect up to the max_limit specified in the service
      config, or the the producer override, if one exists.  The key for this
      map is one of the following:  - '<GROUP_NAME>/<LIMIT_NAME>' for quotas
      defined within quota groups, where GROUP_NAME is the
      google.api.QuotaGroup.name field and LIMIT_NAME is the
      google.api.QuotaLimit.name field from the service config.  For example:
      'ReadGroup/ProjectDaily'.  - '<LIMIT_NAME>' for quotas defined without
      quota groups, where LIMIT_NAME is the google.api.QuotaLimit.name field
      from the service config. For example: 'borrowedCountPerOrganization'.
    effectiveQuotaGroups: Use this field for quota limits defined under quota
      groups. Combines service quota configuration and project-specific
      settings, as a map from quota group name to the effective quota
      information for that group. Output-only.
    effectiveQuotas: The effective quota limits for each group, derived from
      the service defaults together with any producer or consumer overrides.
      For each limit, the effective value is the minimum of the producer and
      consumer overrides if either is present, or else the service default if
      neither is present. DEPRECATED. Use effective_quota_groups instead.
    producerOverrides: Quota overrides set by the producer. Note that if a
      consumer override is also specified, then the minimum of the two will be
      used. This allows consumers to cap their usage voluntarily.  The key for
      this map is one of the following:  - '<GROUP_NAME>/<LIMIT_NAME>' for
      quotas defined within quota groups, where GROUP_NAME is the
      google.api.QuotaGroup.name field and LIMIT_NAME is the
      google.api.QuotaLimit.name field from the service config.  For example:
      'ReadGroup/ProjectDaily'.  - '<LIMIT_NAME>' for quotas defined without
      quota groups, where LIMIT_NAME is the google.api.QuotaLimit.name field
      from the service config. For example: 'borrowedCountPerOrganization'.
    variableTermQuotas: Quotas that are active over a specified time period.
      Only writeable by the producer.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ConsumerOverridesValue(_messages.Message):
        """Quota overrides set by the consumer. Consumer overrides will only have
    an effect up to the max_limit specified in the service config, or the the
    producer override, if one exists.  The key for this map is one of the
    following:  - '<GROUP_NAME>/<LIMIT_NAME>' for quotas defined within quota
    groups, where GROUP_NAME is the google.api.QuotaGroup.name field and
    LIMIT_NAME is the google.api.QuotaLimit.name field from the service
    config.  For example: 'ReadGroup/ProjectDaily'.  - '<LIMIT_NAME>' for
    quotas defined without quota groups, where LIMIT_NAME is the
    google.api.QuotaLimit.name field from the service config. For example:
    'borrowedCountPerOrganization'.

    Messages:
      AdditionalProperty: An additional property for a ConsumerOverridesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        ConsumerOverridesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ConsumerOverridesValue object.

      Fields:
        key: Name of the additional property.
        value: A QuotaLimitOverride attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('QuotaLimitOverride', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EffectiveQuotasValue(_messages.Message):
        """The effective quota limits for each group, derived from the service
    defaults together with any producer or consumer overrides. For each limit,
    the effective value is the minimum of the producer and consumer overrides
    if either is present, or else the service default if neither is present.
    DEPRECATED. Use effective_quota_groups instead.

    Messages:
      AdditionalProperty: An additional property for a EffectiveQuotasValue
        object.

    Fields:
      additionalProperties: Additional properties of type EffectiveQuotasValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a EffectiveQuotasValue object.

      Fields:
        key: Name of the additional property.
        value: A QuotaLimitOverride attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('QuotaLimitOverride', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ProducerOverridesValue(_messages.Message):
        """Quota overrides set by the producer. Note that if a consumer override
    is also specified, then the minimum of the two will be used. This allows
    consumers to cap their usage voluntarily.  The key for this map is one of
    the following:  - '<GROUP_NAME>/<LIMIT_NAME>' for quotas defined within
    quota groups, where GROUP_NAME is the google.api.QuotaGroup.name field and
    LIMIT_NAME is the google.api.QuotaLimit.name field from the service
    config.  For example: 'ReadGroup/ProjectDaily'.  - '<LIMIT_NAME>' for
    quotas defined without quota groups, where LIMIT_NAME is the
    google.api.QuotaLimit.name field from the service config. For example:
    'borrowedCountPerOrganization'.

    Messages:
      AdditionalProperty: An additional property for a ProducerOverridesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        ProducerOverridesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ProducerOverridesValue object.

      Fields:
        key: Name of the additional property.
        value: A QuotaLimitOverride attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('QuotaLimitOverride', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    consumerOverrides = _messages.MessageField('ConsumerOverridesValue', 1)
    effectiveQuotaGroups = _messages.MessageField('EffectiveQuotaGroup', 2, repeated=True)
    effectiveQuotas = _messages.MessageField('EffectiveQuotasValue', 3)
    producerOverrides = _messages.MessageField('ProducerOverridesValue', 4)
    variableTermQuotas = _messages.MessageField('VariableTermQuota', 5, repeated=True)