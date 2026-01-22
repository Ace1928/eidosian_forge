from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsublite.v1 import pubsublite_v1_messages as messages
class TopicStatsProjectsLocationsService(base_api.BaseApiService):
    """Service class for the topicStats_projects_locations resource."""
    _NAME = 'topicStats_projects_locations'

    def __init__(self, client):
        super(PubsubliteV1.TopicStatsProjectsLocationsService, self).__init__(client)
        self._upload_configs = {}