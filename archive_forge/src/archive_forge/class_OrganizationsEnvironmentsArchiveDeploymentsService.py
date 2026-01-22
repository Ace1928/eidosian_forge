from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsArchiveDeploymentsService(base_api.BaseApiService):
    """Service class for the organizations_environments_archiveDeployments resource."""
    _NAME = 'organizations_environments_archiveDeployments'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsArchiveDeploymentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new ArchiveDeployment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsArchiveDeploymentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/archiveDeployments', http_method='POST', method_id='apigee.organizations.environments.archiveDeployments.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/archiveDeployments', request_field='googleCloudApigeeV1ArchiveDeployment', request_type_name='ApigeeOrganizationsEnvironmentsArchiveDeploymentsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an archive deployment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsArchiveDeploymentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/archiveDeployments/{archiveDeploymentsId}', http_method='DELETE', method_id='apigee.organizations.environments.archiveDeployments.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsArchiveDeploymentsDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def GenerateDownloadUrl(self, request, global_params=None):
        """Generates a signed URL for downloading the original zip file used to create an Archive Deployment. The URL is only valid for a limited period and should be used within minutes after generation. Each call returns a new upload URL.

      Args:
        request: (ApigeeOrganizationsEnvironmentsArchiveDeploymentsGenerateDownloadUrlRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1GenerateDownloadUrlResponse) The response message.
      """
        config = self.GetMethodConfig('GenerateDownloadUrl')
        return self._RunMethod(config, request, global_params=global_params)
    GenerateDownloadUrl.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/archiveDeployments/{archiveDeploymentsId}:generateDownloadUrl', http_method='POST', method_id='apigee.organizations.environments.archiveDeployments.generateDownloadUrl', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:generateDownloadUrl', request_field='googleCloudApigeeV1GenerateDownloadUrlRequest', request_type_name='ApigeeOrganizationsEnvironmentsArchiveDeploymentsGenerateDownloadUrlRequest', response_type_name='GoogleCloudApigeeV1GenerateDownloadUrlResponse', supports_download=False)

    def GenerateUploadUrl(self, request, global_params=None):
        """Generates a signed URL for uploading an Archive zip file to Google Cloud Storage. Once the upload is complete, the signed URL should be passed to CreateArchiveDeployment. When uploading to the generated signed URL, please follow these restrictions: * Source file type should be a zip file. * Source file size should not exceed 1GB limit. * No credentials should be attached - the signed URLs provide access to the target bucket using internal service identity; if credentials were attached, the identity from the credentials would be used, but that identity does not have permissions to upload files to the URL. When making a HTTP PUT request, these two headers need to be specified: * `content-type: application/zip` * `x-goog-content-length-range: 0,1073741824` And this header SHOULD NOT be specified: * `Authorization: Bearer YOUR_TOKEN`.

      Args:
        request: (ApigeeOrganizationsEnvironmentsArchiveDeploymentsGenerateUploadUrlRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1GenerateUploadUrlResponse) The response message.
      """
        config = self.GetMethodConfig('GenerateUploadUrl')
        return self._RunMethod(config, request, global_params=global_params)
    GenerateUploadUrl.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/archiveDeployments:generateUploadUrl', http_method='POST', method_id='apigee.organizations.environments.archiveDeployments.generateUploadUrl', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/archiveDeployments:generateUploadUrl', request_field='googleCloudApigeeV1GenerateUploadUrlRequest', request_type_name='ApigeeOrganizationsEnvironmentsArchiveDeploymentsGenerateUploadUrlRequest', response_type_name='GoogleCloudApigeeV1GenerateUploadUrlResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the specified ArchiveDeployment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsArchiveDeploymentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ArchiveDeployment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/archiveDeployments/{archiveDeploymentsId}', http_method='GET', method_id='apigee.organizations.environments.archiveDeployments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsArchiveDeploymentsGetRequest', response_type_name='GoogleCloudApigeeV1ArchiveDeployment', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the ArchiveDeployments in the specified Environment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsArchiveDeploymentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListArchiveDeploymentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/archiveDeployments', http_method='GET', method_id='apigee.organizations.environments.archiveDeployments.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/archiveDeployments', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsArchiveDeploymentsListRequest', response_type_name='GoogleCloudApigeeV1ListArchiveDeploymentsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing ArchiveDeployment. Labels can modified but most of the other fields are not modifiable.

      Args:
        request: (ApigeeOrganizationsEnvironmentsArchiveDeploymentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ArchiveDeployment) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/archiveDeployments/{archiveDeploymentsId}', http_method='PATCH', method_id='apigee.organizations.environments.archiveDeployments.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudApigeeV1ArchiveDeployment', request_type_name='ApigeeOrganizationsEnvironmentsArchiveDeploymentsPatchRequest', response_type_name='GoogleCloudApigeeV1ArchiveDeployment', supports_download=False)