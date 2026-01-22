from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1beta import osconfig_v1beta_messages as messages
class ProjectsZonesService(base_api.BaseApiService):
    """Service class for the projects_zones resource."""
    _NAME = 'projects_zones'

    def __init__(self, client):
        super(OsconfigV1beta.ProjectsZonesService, self).__init__(client)
        self._upload_configs = {}