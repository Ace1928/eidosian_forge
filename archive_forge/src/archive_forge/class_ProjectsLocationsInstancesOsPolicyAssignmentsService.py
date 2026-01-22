from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1alpha import osconfig_v1alpha_messages as messages
class ProjectsLocationsInstancesOsPolicyAssignmentsService(base_api.BaseApiService):
    """Service class for the projects_locations_instances_osPolicyAssignments resource."""
    _NAME = 'projects_locations_instances_osPolicyAssignments'

    def __init__(self, client):
        super(OsconfigV1alpha.ProjectsLocationsInstancesOsPolicyAssignmentsService, self).__init__(client)
        self._upload_configs = {}