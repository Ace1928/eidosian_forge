from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsublite.v1 import pubsublite_v1_messages as messages
class AdminProjectsLocationsService(base_api.BaseApiService):
    """Service class for the admin_projects_locations resource."""
    _NAME = 'admin_projects_locations'

    def __init__(self, client):
        super(PubsubliteV1.AdminProjectsLocationsService, self).__init__(client)
        self._upload_configs = {}