from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1AutoscalingMetricSpec(_messages.Message):
    """The metric specification that defines the target resource utilization
  (CPU utilization, accelerator's duty cycle, and so on) for calculating the
  desired replica count.

  Fields:
    metricName: Required. The resource metric name. Supported metrics: * For
      Online Prediction: *
      `aiplatform.googleapis.com/prediction/online/accelerator/duty_cycle` *
      `aiplatform.googleapis.com/prediction/online/cpu/utilization`
    target: The target resource utilization in percentage (1% - 100%) for the
      given metric; once the real usage deviates from the target by a certain
      percentage, the machine replicas change. The default value is 60
      (representing 60%) if not provided.
  """
    metricName = _messages.StringField(1)
    target = _messages.IntegerField(2, variant=_messages.Variant.INT32)