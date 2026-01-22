from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1 import securitycenter_v1_messages as messages
class OrganizationsSecurityHealthAnalyticsSettingsService(base_api.BaseApiService):
    """Service class for the organizations_securityHealthAnalyticsSettings resource."""
    _NAME = 'organizations_securityHealthAnalyticsSettings'

    def __init__(self, client):
        super(SecuritycenterV1.OrganizationsSecurityHealthAnalyticsSettingsService, self).__init__(client)
        self._upload_configs = {}