from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceusageServicesConsumerQuotaMetricsListRequest(_messages.Message):
    """A ServiceusageServicesConsumerQuotaMetricsListRequest object.

  Enums:
    ViewValueValuesEnum: Specifies the level of detail for quota information
      in the response.

  Fields:
    pageSize: Requested size of the next page of data.
    pageToken: Token identifying which result to start with; returned by a
      previous list call.
    parent: Parent of the quotas resource.  Some example names would be:
      projects/123/services/serviceconsumermanagement.googleapis.com
      folders/345/services/serviceconsumermanagement.googleapis.com
      organizations/456/services/serviceconsumermanagement.googleapis.com
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
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 4)