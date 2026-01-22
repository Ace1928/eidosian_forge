from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.auditmanager.v1alpha import auditmanager_v1alpha_messages as messages
class FoldersLocationsStandardsService(base_api.BaseApiService):
    """Service class for the folders_locations_standards resource."""
    _NAME = 'folders_locations_standards'

    def __init__(self, client):
        super(AuditmanagerV1alpha.FoldersLocationsStandardsService, self).__init__(client)
        self._upload_configs = {}