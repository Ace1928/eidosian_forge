from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v2 import securitycenter_v2_messages as messages
class OrganizationsSourcesLocationsService(base_api.BaseApiService):
    """Service class for the organizations_sources_locations resource."""
    _NAME = 'organizations_sources_locations'

    def __init__(self, client):
        super(SecuritycenterV2.OrganizationsSourcesLocationsService, self).__init__(client)
        self._upload_configs = {}