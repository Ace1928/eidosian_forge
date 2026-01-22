from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackendServiceLogConfig(_messages.Message):
    """The available logging options for the load balancer traffic served by
  this backend service.

  Enums:
    OptionalModeValueValuesEnum: This field can only be specified if logging
      is enabled for this backend service. Configures whether all, none or a
      subset of optional fields should be added to the reported logs. One of
      [INCLUDE_ALL_OPTIONAL, EXCLUDE_ALL_OPTIONAL, CUSTOM]. Default is
      EXCLUDE_ALL_OPTIONAL.

  Fields:
    enable: Denotes whether to enable logging for the load balancer traffic
      served by this backend service. The default value is false.
    optionalFields: This field can only be specified if logging is enabled for
      this backend service and "logConfig.optionalMode" was set to CUSTOM.
      Contains a list of optional fields you want to include in the logs. For
      example: serverInstance, serverGkeDetails.cluster,
      serverGkeDetails.pod.podNamespace
    optionalMode: This field can only be specified if logging is enabled for
      this backend service. Configures whether all, none or a subset of
      optional fields should be added to the reported logs. One of
      [INCLUDE_ALL_OPTIONAL, EXCLUDE_ALL_OPTIONAL, CUSTOM]. Default is
      EXCLUDE_ALL_OPTIONAL.
    sampleRate: This field can only be specified if logging is enabled for
      this backend service. The value of the field must be in [0, 1]. This
      configures the sampling rate of requests to the load balancer where 1.0
      means all logged requests are reported and 0.0 means no logged requests
      are reported. The default value is 1.0.
  """

    class OptionalModeValueValuesEnum(_messages.Enum):
        """This field can only be specified if logging is enabled for this
    backend service. Configures whether all, none or a subset of optional
    fields should be added to the reported logs. One of [INCLUDE_ALL_OPTIONAL,
    EXCLUDE_ALL_OPTIONAL, CUSTOM]. Default is EXCLUDE_ALL_OPTIONAL.

    Values:
      CUSTOM: A subset of optional fields.
      EXCLUDE_ALL_OPTIONAL: None optional fields.
      INCLUDE_ALL_OPTIONAL: All optional fields.
    """
        CUSTOM = 0
        EXCLUDE_ALL_OPTIONAL = 1
        INCLUDE_ALL_OPTIONAL = 2
    enable = _messages.BooleanField(1)
    optionalFields = _messages.StringField(2, repeated=True)
    optionalMode = _messages.EnumField('OptionalModeValueValuesEnum', 3)
    sampleRate = _messages.FloatField(4, variant=_messages.Variant.FLOAT)