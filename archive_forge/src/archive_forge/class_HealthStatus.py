from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthStatus(_messages.Message):
    """A HealthStatus object.

  Enums:
    HealthStateValueValuesEnum: Health state of the IPv4 address of the
      instance.
    Ipv6HealthStateValueValuesEnum: Health state of the IPv6 address of the
      instance.
    WeightErrorValueValuesEnum:

  Messages:
    AnnotationsValue: Metadata defined as annotations for network endpoint.

  Fields:
    annotations: Metadata defined as annotations for network endpoint.
    forwardingRule: URL of the forwarding rule associated with the health
      status of the instance.
    forwardingRuleIp: A forwarding rule IP address assigned to this instance.
    healthState: Health state of the IPv4 address of the instance.
    instance: URL of the instance resource.
    ipAddress: For target pool based Network Load Balancing, it indicates the
      forwarding rule's IP address assigned to this instance. For other types
      of load balancing, the field indicates VM internal ip.
    ipv6Address: A string attribute.
    ipv6HealthState: Health state of the IPv6 address of the instance.
    port: The named port of the instance group, not necessarily the port that
      is health-checked.
    weight: A string attribute.
    weightError: A WeightErrorValueValuesEnum attribute.
  """

    class HealthStateValueValuesEnum(_messages.Enum):
        """Health state of the IPv4 address of the instance.

    Values:
      HEALTHY: <no description>
      UNHEALTHY: <no description>
    """
        HEALTHY = 0
        UNHEALTHY = 1

    class Ipv6HealthStateValueValuesEnum(_messages.Enum):
        """Health state of the IPv6 address of the instance.

    Values:
      HEALTHY: <no description>
      UNHEALTHY: <no description>
    """
        HEALTHY = 0
        UNHEALTHY = 1

    class WeightErrorValueValuesEnum(_messages.Enum):
        """WeightErrorValueValuesEnum enum type.

    Values:
      INVALID_WEIGHT: The response to a Health Check probe had the HTTP
        response header field X-Load-Balancing-Endpoint-Weight, but its
        content was invalid (i.e., not a non-negative single-precision
        floating-point number in decimal string representation).
      MISSING_WEIGHT: The response to a Health Check probe did not have the
        HTTP response header field X-Load-Balancing-Endpoint-Weight.
      UNAVAILABLE_WEIGHT: This is the value when the accompanied health status
        is either TIMEOUT (i.e.,the Health Check probe was not able to get a
        response in time) or UNKNOWN. For the latter, it should be typically
        because there has not been sufficient time to parse and report the
        weight for a new backend (which is with 0.0.0.0 ip address). However,
        it can be also due to an outage case for which the health status is
        explicitly reset to UNKNOWN.
      WEIGHT_NONE: This is the default value when WeightReportMode is DISABLE,
        and is also the initial value when WeightReportMode has just updated
        to ENABLE or DRY_RUN and there has not been sufficient time to parse
        and report the backend weight.
    """
        INVALID_WEIGHT = 0
        MISSING_WEIGHT = 1
        UNAVAILABLE_WEIGHT = 2
        WEIGHT_NONE = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Metadata defined as annotations for network endpoint.

    Messages:
      AdditionalProperty: An additional property for a AnnotationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AnnotationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    annotations = _messages.MessageField('AnnotationsValue', 1)
    forwardingRule = _messages.StringField(2)
    forwardingRuleIp = _messages.StringField(3)
    healthState = _messages.EnumField('HealthStateValueValuesEnum', 4)
    instance = _messages.StringField(5)
    ipAddress = _messages.StringField(6)
    ipv6Address = _messages.StringField(7)
    ipv6HealthState = _messages.EnumField('Ipv6HealthStateValueValuesEnum', 8)
    port = _messages.IntegerField(9, variant=_messages.Variant.INT32)
    weight = _messages.StringField(10)
    weightError = _messages.EnumField('WeightErrorValueValuesEnum', 11)