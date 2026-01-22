from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingOrganizationsLogsListRequest(_messages.Message):
    """A LoggingOrganizationsLogsListRequest object.

  Fields:
    pageSize: Optional. The maximum number of results to return from this
      request. Non-positive values are ignored. The presence of nextPageToken
      in the response indicates that more results might be available.
    pageToken: Optional. If present, then retrieve the next batch of results
      from the preceding call to this method. pageToken must be the value of
      nextPageToken from the previous response. The values of other method
      parameters should be identical to those in the previous call.
    parent: Required. The resource name to list logs for:
      projects/[PROJECT_ID] organizations/[ORGANIZATION_ID]
      billingAccounts/[BILLING_ACCOUNT_ID] folders/[FOLDER_ID]
    resourceNames: Optional. List of resource names to list logs for: projects
      /[PROJECT_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]/views/[VIEW_ID
      ] organizations/[ORGANIZATION_ID]/locations/[LOCATION_ID]/buckets/[BUCKE
      T_ID]/views/[VIEW_ID] billingAccounts/[BILLING_ACCOUNT_ID]/locations/[LO
      CATION_ID]/buckets/[BUCKET_ID]/views/[VIEW_ID] folders/[FOLDER_ID]/locat
      ions/[LOCATION_ID]/buckets/[BUCKET_ID]/views/[VIEW_ID]To support legacy
      queries, it could also be: projects/[PROJECT_ID]
      organizations/[ORGANIZATION_ID] billingAccounts/[BILLING_ACCOUNT_ID]
      folders/[FOLDER_ID]The resource name in the parent field is added to
      this list.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    resourceNames = _messages.StringField(4, repeated=True)