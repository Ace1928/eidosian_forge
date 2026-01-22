from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recommender.v1alpha2 import recommender_v1alpha2_messages as messages
def MarkActive(self, request, global_params=None):
    """Mark the Recommendation State as Active. Users can use this method to indicate to the Recommender API that a DISMISSED recommendation has to be marked back as ACTIVE. MarkRecommendationActive can be applied to recommendations in DISMISSED state. Requires the recommender.*.update IAM permission for the specified recommender.

      Args:
        request: (RecommenderProjectsLocationsRecommendersRecommendationsMarkActiveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecommenderV1alpha2Recommendation) The response message.
      """
    config = self.GetMethodConfig('MarkActive')
    return self._RunMethod(config, request, global_params=global_params)