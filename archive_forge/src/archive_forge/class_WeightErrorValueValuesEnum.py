from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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