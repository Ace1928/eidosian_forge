from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceconsumermanagementServicesConsumerQuotaMetricsGetRequest(_messages.Message):
    """A ServiceconsumermanagementServicesConsumerQuotaMetricsGetRequest
  object.

  Enums:
    ViewValueValuesEnum: Specifies the level of detail for quota information
      in the response.

  Fields:
    name: The resource name of the quota metric, returned by a
      ListConsumerQuotaMetrics call.  An example name would be: `services/comp
      ute.googleapis.com/projects/123/consumerQuotaMetrics/compute.googleapis.
      com%2Fcpus`
    view: Specifies the level of detail for quota information in the response.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Specifies the level of detail for quota information in the response.

    Values:
      QUOTA_VIEW_UNSPECIFIED: <no description>
      BASIC: <no description>
      FULL: <no description>
    """
        QUOTA_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)