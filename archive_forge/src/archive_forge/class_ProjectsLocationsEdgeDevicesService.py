from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
class ProjectsLocationsEdgeDevicesService(base_api.BaseApiService):
    """Service class for the projects_locations_edgeDevices resource."""
    _NAME = 'projects_locations_edgeDevices'

    def __init__(self, client):
        super(AiplatformV1beta1.ProjectsLocationsEdgeDevicesService, self).__init__(client)
        self._upload_configs = {}