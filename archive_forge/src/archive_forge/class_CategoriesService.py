from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.serviceusage.v2alpha import serviceusage_v2alpha_messages as messages
class CategoriesService(base_api.BaseApiService):
    """Service class for the categories resource."""
    _NAME = 'categories'

    def __init__(self, client):
        super(ServiceusageV2alpha.CategoriesService, self).__init__(client)
        self._upload_configs = {}