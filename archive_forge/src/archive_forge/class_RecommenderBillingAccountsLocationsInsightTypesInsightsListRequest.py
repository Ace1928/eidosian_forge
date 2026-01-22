from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecommenderBillingAccountsLocationsInsightTypesInsightsListRequest(_messages.Message):
    """A RecommenderBillingAccountsLocationsInsightTypesInsightsListRequest
  object.

  Fields:
    filter: Filter expression to restrict the insights returned. Supported
      filter fields: * `stateInfo.state` * `insightSubtype` * `severity` *
      `targetResources` Examples: * `stateInfo.state = ACTIVE OR
      stateInfo.state = DISMISSED` * `insightSubtype = PERMISSIONS_USAGE` *
      `severity = CRITICAL OR severity = HIGH` * `targetResources :
      //compute.googleapis.com/projects/1234/zones/us-
      central1-a/instances/instance-1` * `stateInfo.state = ACTIVE AND
      (severity = CRITICAL OR severity = HIGH)` The max allowed filter length
      is 500 characters. (These expressions are based on the filter language
      described at https://google.aip.dev/160)
    pageSize: Optional. The maximum number of results to return from this
      request. Non-positive values are ignored. If not specified, the server
      will determine the number of results to return.
    pageToken: Optional. If present, retrieves the next batch of results from
      the preceding call to this method. `page_token` must be the value of
      `next_page_token` from the previous response. The values of other method
      parameters must be identical to those in the previous call.
    parent: Required. The container resource on which to execute the request.
      Acceptable formats: * `projects/[PROJECT_NUMBER]/locations/[LOCATION]/in
      sightTypes/[INSIGHT_TYPE_ID]` * `projects/[PROJECT_ID]/locations/[LOCATI
      ON]/insightTypes/[INSIGHT_TYPE_ID]` * `billingAccounts/[BILLING_ACCOUNT_
      ID]/locations/[LOCATION]/insightTypes/[INSIGHT_TYPE_ID]` * `folders/[FOL
      DER_ID]/locations/[LOCATION]/insightTypes/[INSIGHT_TYPE_ID]` * `organiza
      tions/[ORGANIZATION_ID]/locations/[LOCATION]/insightTypes/[INSIGHT_TYPE_
      ID]` LOCATION here refers to GCP Locations:
      https://cloud.google.com/about/locations/ INSIGHT_TYPE_ID refers to
      supported insight types:
      https://cloud.google.com/recommender/docs/insights/insight-types.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)