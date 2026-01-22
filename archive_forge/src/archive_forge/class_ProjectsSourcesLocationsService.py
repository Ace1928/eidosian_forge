from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v2 import securitycenter_v2_messages as messages
class ProjectsSourcesLocationsService(base_api.BaseApiService):
    """Service class for the projects_sources_locations resource."""
    _NAME = 'projects_sources_locations'

    def __init__(self, client):
        super(SecuritycenterV2.ProjectsSourcesLocationsService, self).__init__(client)
        self._upload_configs = {}