from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsPublishersService(base_api.BaseApiService):
    """Service class for the projects_locations_publishers resource."""
    _NAME = 'projects_locations_publishers'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsPublishersService, self).__init__(client)
        self._upload_configs = {}