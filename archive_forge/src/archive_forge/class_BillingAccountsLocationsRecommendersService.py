from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recommender.v1alpha2 import recommender_v1alpha2_messages as messages
class BillingAccountsLocationsRecommendersService(base_api.BaseApiService):
    """Service class for the billingAccounts_locations_recommenders resource."""
    _NAME = 'billingAccounts_locations_recommenders'

    def __init__(self, client):
        super(RecommenderV1alpha2.BillingAccountsLocationsRecommendersService, self).__init__(client)
        self._upload_configs = {}