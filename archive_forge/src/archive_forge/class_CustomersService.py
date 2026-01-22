from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1 import cloudidentity_v1_messages as messages
class CustomersService(base_api.BaseApiService):
    """Service class for the customers resource."""
    _NAME = 'customers'

    def __init__(self, client):
        super(CloudidentityV1.CustomersService, self).__init__(client)
        self._upload_configs = {}