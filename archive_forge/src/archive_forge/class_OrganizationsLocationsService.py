from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securedlandingzone.v1beta import securedlandingzone_v1beta_messages as messages
class OrganizationsLocationsService(base_api.BaseApiService):
    """Service class for the organizations_locations resource."""
    _NAME = 'organizations_locations'

    def __init__(self, client):
        super(SecuredlandingzoneV1beta.OrganizationsLocationsService, self).__init__(client)
        self._upload_configs = {}