from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
class ProjectsLocationsGithubService(base_api.BaseApiService):
    """Service class for the projects_locations_github resource."""
    _NAME = 'projects_locations_github'

    def __init__(self, client):
        super(CloudbuildV1.ProjectsLocationsGithubService, self).__init__(client)
        self._upload_configs = {}