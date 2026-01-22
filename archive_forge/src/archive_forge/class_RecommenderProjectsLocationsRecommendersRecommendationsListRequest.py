from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecommenderProjectsLocationsRecommendersRecommendationsListRequest(_messages.Message):
    """A RecommenderProjectsLocationsRecommendersRecommendationsListRequest
  object.

  Fields:
    filter: Filter expression to restrict the recommendations returned.
      Supported filter fields: * `state_info.state` * `recommenderSubtype` *
      `priority` * `targetResources` Examples: * `stateInfo.state = ACTIVE OR
      stateInfo.state = DISMISSED` * `recommenderSubtype = REMOVE_ROLE OR
      recommenderSubtype = REPLACE_ROLE` * `priority = P1 OR priority = P2` *
      `targetResources : //compute.googleapis.com/projects/1234/zones/us-
      central1-a/instances/instance-1` * `stateInfo.state = ACTIVE AND
      (priority = P1 OR priority = P2)` The max allowed filter length is 500
      characters. (These expressions are based on the filter language
      described at https://google.aip.dev/160)
    pageSize: Optional. The maximum number of results to return from this
      request. Non-positive values are ignored. If not specified, the server
      will determine the number of results to return.
    pageToken: Optional. If present, retrieves the next batch of results from
      the preceding call to this method. `page_token` must be the value of
      `next_page_token` from the previous response. The values of other method
      parameters must be identical to those in the previous call.
    parent: Required. The container resource on which to execute the request.
      Acceptable formats: * `projects/[PROJECT_NUMBER]/locations/[LOCATION]/re
      commenders/[RECOMMENDER_ID]` * `projects/[PROJECT_ID]/locations/[LOCATIO
      N]/recommenders/[RECOMMENDER_ID]` * `billingAccounts/[BILLING_ACCOUNT_ID
      ]/locations/[LOCATION]/recommenders/[RECOMMENDER_ID]` *
      `folders/[FOLDER_ID]/locations/[LOCATION]/recommenders/[RECOMMENDER_ID]`
      * `organizations/[ORGANIZATION_ID]/locations/[LOCATION]/recommenders/[RE
      COMMENDER_ID]` LOCATION here refers to GCP Locations:
      https://cloud.google.com/about/locations/ RECOMMENDER_ID refers to
      supported recommenders:
      https://cloud.google.com/recommender/docs/recommenders.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)