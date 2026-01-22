from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OutlierDetection(_messages.Message):
    """Settings controlling the eviction of unhealthy hosts from the load
  balancing pool for the backend service.

  Fields:
    baseEjectionTime: The base time that a backend endpoint is ejected for.
      Defaults to 30000ms or 30s. After a backend endpoint is returned back to
      the load balancing pool, it can be ejected again in another ejection
      analysis. Thus, the total ejection time is equal to the base ejection
      time multiplied by the number of times the backend endpoint has been
      ejected. Defaults to 30000ms or 30s.
    consecutiveErrors: Number of consecutive errors before a backend endpoint
      is ejected from the load balancing pool. When the backend endpoint is
      accessed over HTTP, a 5xx return code qualifies as an error. Defaults to
      5.
    consecutiveGatewayFailure: The number of consecutive gateway failures
      (502, 503, 504 status or connection errors that are mapped to one of
      those status codes) before a consecutive gateway failure ejection
      occurs. Defaults to 3.
    enforcingConsecutiveErrors: The percentage chance that a backend endpoint
      will be ejected when an outlier status is detected through consecutive
      5xx. This setting can be used to disable ejection or to ramp it up
      slowly. Defaults to 0.
    enforcingConsecutiveGatewayFailure: The percentage chance that a backend
      endpoint will be ejected when an outlier status is detected through
      consecutive gateway failures. This setting can be used to disable
      ejection or to ramp it up slowly. Defaults to 100.
    enforcingSuccessRate: The percentage chance that a backend endpoint will
      be ejected when an outlier status is detected through success rate
      statistics. This setting can be used to disable ejection or to ramp it
      up slowly. Defaults to 100. Not supported when the backend service uses
      Serverless NEG.
    interval: Time interval between ejection analysis sweeps. This can result
      in both new ejections and backend endpoints being returned to service.
      The interval is equal to the number of seconds as defined in
      outlierDetection.interval.seconds plus the number of nanoseconds as
      defined in outlierDetection.interval.nanos. Defaults to 1 second.
    maxEjectionPercent: Maximum percentage of backend endpoints in the load
      balancing pool for the backend service that can be ejected if the
      ejection conditions are met. Defaults to 50%.
    successRateMinimumHosts: The number of backend endpoints in the load
      balancing pool that must have enough request volume to detect success
      rate outliers. If the number of backend endpoints is fewer than this
      setting, outlier detection via success rate statistics is not performed
      for any backend endpoint in the load balancing pool. Defaults to 5. Not
      supported when the backend service uses Serverless NEG.
    successRateRequestVolume: The minimum number of total requests that must
      be collected in one interval (as defined by the interval duration above)
      to include this backend endpoint in success rate based outlier
      detection. If the volume is lower than this setting, outlier detection
      via success rate statistics is not performed for that backend endpoint.
      Defaults to 100. Not supported when the backend service uses Serverless
      NEG.
    successRateStdevFactor: This factor is used to determine the ejection
      threshold for success rate outlier ejection. The ejection threshold is
      the difference between the mean success rate, and the product of this
      factor and the standard deviation of the mean success rate: mean -
      (stdev * successRateStdevFactor). This factor is divided by a thousand
      to get a double. That is, if the desired factor is 1.9, the runtime
      value should be 1900. Defaults to 1900. Not supported when the backend
      service uses Serverless NEG.
  """
    baseEjectionTime = _messages.MessageField('Duration', 1)
    consecutiveErrors = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    consecutiveGatewayFailure = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    enforcingConsecutiveErrors = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    enforcingConsecutiveGatewayFailure = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    enforcingSuccessRate = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    interval = _messages.MessageField('Duration', 7)
    maxEjectionPercent = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    successRateMinimumHosts = _messages.IntegerField(9, variant=_messages.Variant.INT32)
    successRateRequestVolume = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    successRateStdevFactor = _messages.IntegerField(11, variant=_messages.Variant.INT32)