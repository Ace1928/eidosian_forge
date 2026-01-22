from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.servicenetworking.v1 import servicenetworking_v1_messages as messages
class ServicesProjectsService(base_api.BaseApiService):
    """Service class for the services_projects resource."""
    _NAME = 'services_projects'

    def __init__(self, client):
        super(ServicenetworkingV1.ServicesProjectsService, self).__init__(client)
        self._upload_configs = {}