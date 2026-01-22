from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2ResourceRequirements(_messages.Message):
    """ResourceRequirements describes the compute resource requirements.

  Messages:
    LimitsValue: Only `memory` and `cpu` keys in the map are supported. Notes:
      * The only supported values for CPU are '1', '2', '4', and '8'. Setting
      4 CPU requires at least 2Gi of memory. For more information, go to
      https://cloud.google.com/run/docs/configuring/cpu. * For supported
      'memory' values and syntax, go to
      https://cloud.google.com/run/docs/configuring/memory-limits

  Fields:
    cpuIdle: Determines whether CPU is only allocated during requests (true by
      default). However, if ResourceRequirements is set, the caller must
      explicitly set this field to true to preserve the default behavior.
    limits: Only `memory` and `cpu` keys in the map are supported. Notes: *
      The only supported values for CPU are '1', '2', '4', and '8'. Setting 4
      CPU requires at least 2Gi of memory. For more information, go to
      https://cloud.google.com/run/docs/configuring/cpu. * For supported
      'memory' values and syntax, go to
      https://cloud.google.com/run/docs/configuring/memory-limits
    startupCpuBoost: Determines whether CPU should be boosted on startup of a
      new container instance above the requested CPU threshold, this can help
      reduce cold-start latency.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LimitsValue(_messages.Message):
        """Only `memory` and `cpu` keys in the map are supported. Notes: * The
    only supported values for CPU are '1', '2', '4', and '8'. Setting 4 CPU
    requires at least 2Gi of memory. For more information, go to
    https://cloud.google.com/run/docs/configuring/cpu. * For supported
    'memory' values and syntax, go to
    https://cloud.google.com/run/docs/configuring/memory-limits

    Messages:
      AdditionalProperty: An additional property for a LimitsValue object.

    Fields:
      additionalProperties: Additional properties of type LimitsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LimitsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    cpuIdle = _messages.BooleanField(1)
    limits = _messages.MessageField('LimitsValue', 2)
    startupCpuBoost = _messages.BooleanField(3)