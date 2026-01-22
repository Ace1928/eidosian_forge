from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListLogEntriesRequest(_messages.Message):
    """The parameters to ListLogEntries.

  Fields:
    filter: Optional. A filter that chooses which log entries to return. For
      more information, see Logging query language
      (https://cloud.google.com/logging/docs/view/logging-query-language).Only
      log entries that match the filter are returned. An empty filter matches
      all log entries in the resources listed in resource_names. Referencing a
      parent resource that is not listed in resource_names will cause the
      filter to return no results. The maximum length of a filter is 20,000
      characters.
    orderBy: Optional. How the results should be sorted. Presently, the only
      permitted values are "timestamp asc" (default) and "timestamp desc". The
      first option returns entries in order of increasing values of
      LogEntry.timestamp (oldest first), and the second option returns entries
      in order of decreasing timestamps (newest first). Entries with equal
      timestamps are returned in order of their insert_id values.
    pageSize: Optional. The maximum number of results to return from this
      request. Default is 50. If the value is negative, the request is
      rejected.The presence of next_page_token in the response indicates that
      more results might be available.
    pageToken: Optional. If present, then retrieve the next batch of results
      from the preceding call to this method. page_token must be the value of
      next_page_token from the previous response. The values of other method
      parameters should be identical to those in the previous call.
    projectIds: Optional. Deprecated. Use resource_names instead. One or more
      project identifiers or project numbers from which to retrieve log
      entries. Example: "my-project-1A".
    resourceNames: Required. Names of one or more parent resources from which
      to retrieve log entries: projects/[PROJECT_ID]
      organizations/[ORGANIZATION_ID] billingAccounts/[BILLING_ACCOUNT_ID]
      folders/[FOLDER_ID]May alternatively be one or more views: projects/[PRO
      JECT_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]/views/[VIEW_ID] org
      anizations/[ORGANIZATION_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]
      /views/[VIEW_ID] billingAccounts/[BILLING_ACCOUNT_ID]/locations/[LOCATIO
      N_ID]/buckets/[BUCKET_ID]/views/[VIEW_ID] folders/[FOLDER_ID]/locations/
      [LOCATION_ID]/buckets/[BUCKET_ID]/views/[VIEW_ID]Projects listed in the
      project_ids field are added to this list. A maximum of 100 resources may
      be specified in a single request.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    projectIds = _messages.StringField(5, repeated=True)
    resourceNames = _messages.StringField(6, repeated=True)