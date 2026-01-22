from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recommender.v1alpha2 import recommender_v1alpha2_messages as messages
class OrganizationsLocationsInsightTypesService(base_api.BaseApiService):
    """Service class for the organizations_locations_insightTypes resource."""
    _NAME = 'organizations_locations_insightTypes'

    def __init__(self, client):
        super(RecommenderV1alpha2.OrganizationsLocationsInsightTypesService, self).__init__(client)
        self._upload_configs = {}