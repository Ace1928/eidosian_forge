from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.container.v1 import container_v1_messages as messages
class ProjectsAggregatedService(base_api.BaseApiService):
    """Service class for the projects_aggregated resource."""
    _NAME = 'projects_aggregated'

    def __init__(self, client):
        super(ContainerV1.ProjectsAggregatedService, self).__init__(client)
        self._upload_configs = {}