from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v1 import monitoring_v1_messages as messages
class ProjectsLocationService(base_api.BaseApiService):
    """Service class for the projects_location resource."""
    _NAME = 'projects_location'

    def __init__(self, client):
        super(MonitoringV1.ProjectsLocationService, self).__init__(client)
        self._upload_configs = {}