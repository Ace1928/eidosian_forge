from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v1 import monitoring_v1_messages as messages
class ProjectsLocationPrometheusApiService(base_api.BaseApiService):
    """Service class for the projects_location_prometheus_api resource."""
    _NAME = 'projects_location_prometheus_api'

    def __init__(self, client):
        super(MonitoringV1.ProjectsLocationPrometheusApiService, self).__init__(client)
        self._upload_configs = {}