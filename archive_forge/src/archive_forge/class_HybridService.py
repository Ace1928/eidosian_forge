from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class HybridService(base_api.BaseApiService):
    """Service class for the hybrid resource."""
    _NAME = 'hybrid'

    def __init__(self, client):
        super(ApigeeV1.HybridService, self).__init__(client)
        self._upload_configs = {}