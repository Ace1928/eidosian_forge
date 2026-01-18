from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recommender.v1 import recommender_v1_messages as messages
Updates a Recommender Config. This will create a new revision of the config.

      Args:
        request: (RecommenderProjectsLocationsRecommendersUpdateConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecommenderV1RecommenderConfig) The response message.
      