from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsublite.v1 import pubsublite_v1_messages as messages
class CursorService(base_api.BaseApiService):
    """Service class for the cursor resource."""
    _NAME = 'cursor'

    def __init__(self, client):
        super(PubsubliteV1.CursorService, self).__init__(client)
        self._upload_configs = {}