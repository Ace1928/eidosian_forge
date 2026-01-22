from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha2 import gkehub_v1alpha2_messages as messages
class ProjectsLocationsGlobalMembershipsService(base_api.BaseApiService):
    """Service class for the projects_locations_global_memberships resource."""
    _NAME = 'projects_locations_global_memberships'

    def __init__(self, client):
        super(GkehubV1alpha2.ProjectsLocationsGlobalMembershipsService, self).__init__(client)
        self._upload_configs = {}

    def InitializeHub(self, request, global_params=None):
        """Initializes the Hub in this project, which includes creating the default Hub Service Account and the Hub Workload Identity Pool. Initialization is optional, and happens automatically when the first Membership is created. InitializeHub should be called when the first Membership cannot be registered without these resources. A common example is granting the Hub Service Account access to another project, which requires the account to exist first.

      Args:
        request: (GkehubProjectsLocationsGlobalMembershipsInitializeHubRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InitializeHubResponse) The response message.
      """
        config = self.GetMethodConfig('InitializeHub')
        return self._RunMethod(config, request, global_params=global_params)
    InitializeHub.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/global/memberships:initializeHub', http_method='POST', method_id='gkehub.projects.locations.global.memberships.initializeHub', ordered_params=['project'], path_params=['project'], query_params=[], relative_path='v1alpha2/{+project}:initializeHub', request_field='initializeHubRequest', request_type_name='GkehubProjectsLocationsGlobalMembershipsInitializeHubRequest', response_type_name='InitializeHubResponse', supports_download=False)