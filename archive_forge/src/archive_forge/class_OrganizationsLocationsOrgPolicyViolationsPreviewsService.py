from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.policysimulator.v1beta import policysimulator_v1beta_messages as messages
class OrganizationsLocationsOrgPolicyViolationsPreviewsService(base_api.BaseApiService):
    """Service class for the organizations_locations_orgPolicyViolationsPreviews resource."""
    _NAME = 'organizations_locations_orgPolicyViolationsPreviews'

    def __init__(self, client):
        super(PolicysimulatorV1beta.OrganizationsLocationsOrgPolicyViolationsPreviewsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """CreateOrgPolicyViolationsPreview creates an OrgPolicyViolationsPreview for the proposed changes in the provided OrgPolicyViolationsPreview.OrgPolicyOverlay. The changes to OrgPolicy are specified by this `OrgPolicyOverlay`. The resources to scan are inferred from these specified changes.

      Args:
        request: (PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/organizations/{organizationsId}/locations/{locationsId}/orgPolicyViolationsPreviews', http_method='POST', method_id='policysimulator.organizations.locations.orgPolicyViolationsPreviews.create', ordered_params=['parent'], path_params=['parent'], query_params=['orgPolicyViolationsPreviewId'], relative_path='v1beta/{+parent}/orgPolicyViolationsPreviews', request_field='googleCloudPolicysimulatorV1betaOrgPolicyViolationsPreview', request_type_name='PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Generate(self, request, global_params=None):
        """GenerateOrgPolicyViolationsPreview generates an OrgPolicyViolationsPreview for the proposed changes in the provided OrgPolicyViolationsPreview.OrgPolicyOverlay. The changes to OrgPolicy are specified by this `OrgPolicyOverlay`. The resources to scan are inferred from these specified changes.

      Args:
        request: (PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsGenerateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Generate')
        return self._RunMethod(config, request, global_params=global_params)
    Generate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/organizations/{organizationsId}/locations/{locationsId}/orgPolicyViolationsPreviews:generate', http_method='POST', method_id='policysimulator.organizations.locations.orgPolicyViolationsPreviews.generate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1beta/{+parent}/orgPolicyViolationsPreviews:generate', request_field='googleCloudPolicysimulatorV1betaOrgPolicyViolationsPreview', request_type_name='PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsGenerateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """GetOrgPolicyViolationsPreview gets the specified OrgPolicyViolationsPreview. Each OrgPolicyViolationsPreview is available for at least 7 days.

      Args:
        request: (PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudPolicysimulatorV1betaOrgPolicyViolationsPreview) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/organizations/{organizationsId}/locations/{locationsId}/orgPolicyViolationsPreviews/{orgPolicyViolationsPreviewsId}', http_method='GET', method_id='policysimulator.organizations.locations.orgPolicyViolationsPreviews.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsGetRequest', response_type_name='GoogleCloudPolicysimulatorV1betaOrgPolicyViolationsPreview', supports_download=False)

    def List(self, request, global_params=None):
        """ListOrgPolicyViolationsPreviews lists each OrgPolicyViolationsPreview in an organization. Each OrgPolicyViolationsPreview is available for at least 7 days.

      Args:
        request: (PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudPolicysimulatorV1betaListOrgPolicyViolationsPreviewsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/organizations/{organizationsId}/locations/{locationsId}/orgPolicyViolationsPreviews', http_method='GET', method_id='policysimulator.organizations.locations.orgPolicyViolationsPreviews.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/orgPolicyViolationsPreviews', request_field='', request_type_name='PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsListRequest', response_type_name='GoogleCloudPolicysimulatorV1betaListOrgPolicyViolationsPreviewsResponse', supports_download=False)