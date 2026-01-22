from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.batch.v1alpha import batch_v1alpha_messages as messages
class ProjectsLocationsJobsTaskGroupsService(base_api.BaseApiService):
    """Service class for the projects_locations_jobs_taskGroups resource."""
    _NAME = 'projects_locations_jobs_taskGroups'

    def __init__(self, client):
        super(BatchV1alpha.ProjectsLocationsJobsTaskGroupsService, self).__init__(client)
        self._upload_configs = {}