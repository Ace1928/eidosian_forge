from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.workflowexecutions.v1beta import workflowexecutions_v1beta_messages as messages
class ProjectsLocationsWorkflowsService(base_api.BaseApiService):
    """Service class for the projects_locations_workflows resource."""
    _NAME = 'projects_locations_workflows'

    def __init__(self, client):
        super(WorkflowexecutionsV1beta.ProjectsLocationsWorkflowsService, self).__init__(client)
        self._upload_configs = {}