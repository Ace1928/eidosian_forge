from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.policyanalyzer.v1 import policyanalyzer_v1_messages as messages
class ProjectsLocationsActivityTypesService(base_api.BaseApiService):
    """Service class for the projects_locations_activityTypes resource."""
    _NAME = 'projects_locations_activityTypes'

    def __init__(self, client):
        super(PolicyanalyzerV1.ProjectsLocationsActivityTypesService, self).__init__(client)
        self._upload_configs = {}