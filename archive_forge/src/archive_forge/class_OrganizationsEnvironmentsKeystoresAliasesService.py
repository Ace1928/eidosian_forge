from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsKeystoresAliasesService(base_api.BaseApiService):
    """Service class for the organizations_environments_keystores_aliases resource."""
    _NAME = 'organizations_environments_keystores_aliases'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsKeystoresAliasesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an alias from a key/certificate pair. The structure of the request is controlled by the `format` query parameter: - `keycertfile` - Separate PEM-encoded key and certificate files are uploaded. Set `Content-Type: multipart/form-data` and include the `keyFile`, `certFile`, and `password` (if keys are encrypted) fields in the request body. If uploading to a truststore, omit `keyFile`. - `pkcs12` - A PKCS12 file is uploaded. Set `Content-Type: multipart/form-data`, provide the file in the `file` field, and include the `password` field if the file is encrypted in the request body. - `selfsignedcert` - A new private key and certificate are generated. Set `Content-Type: application/json` and include CertificateGenerationSpec in the request body.

      Args:
        request: (ApigeeOrganizationsEnvironmentsKeystoresAliasesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Alias) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/keystores/{keystoresId}/aliases', http_method='POST', method_id='apigee.organizations.environments.keystores.aliases.create', ordered_params=['parent'], path_params=['parent'], query_params=['_password', 'alias', 'format', 'ignoreExpiryValidation', 'ignoreNewlineValidation'], relative_path='v1/{+parent}/aliases', request_field='googleApiHttpBody', request_type_name='ApigeeOrganizationsEnvironmentsKeystoresAliasesCreateRequest', response_type_name='GoogleCloudApigeeV1Alias', supports_download=False)

    def Csr(self, request, global_params=None):
        """Generates a PKCS #10 Certificate Signing Request for the private key in an alias.

      Args:
        request: (ApigeeOrganizationsEnvironmentsKeystoresAliasesCsrRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleApiHttpBody) The response message.
      """
        config = self.GetMethodConfig('Csr')
        return self._RunMethod(config, request, global_params=global_params)
    Csr.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/keystores/{keystoresId}/aliases/{aliasesId}/csr', http_method='GET', method_id='apigee.organizations.environments.keystores.aliases.csr', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}/csr', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsKeystoresAliasesCsrRequest', response_type_name='GoogleApiHttpBody', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an alias.

      Args:
        request: (ApigeeOrganizationsEnvironmentsKeystoresAliasesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Alias) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/keystores/{keystoresId}/aliases/{aliasesId}', http_method='DELETE', method_id='apigee.organizations.environments.keystores.aliases.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsKeystoresAliasesDeleteRequest', response_type_name='GoogleCloudApigeeV1Alias', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an alias.

      Args:
        request: (ApigeeOrganizationsEnvironmentsKeystoresAliasesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Alias) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/keystores/{keystoresId}/aliases/{aliasesId}', http_method='GET', method_id='apigee.organizations.environments.keystores.aliases.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsKeystoresAliasesGetRequest', response_type_name='GoogleCloudApigeeV1Alias', supports_download=False)

    def GetCertificate(self, request, global_params=None):
        """Gets the certificate from an alias in PEM-encoded form.

      Args:
        request: (ApigeeOrganizationsEnvironmentsKeystoresAliasesGetCertificateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleApiHttpBody) The response message.
      """
        config = self.GetMethodConfig('GetCertificate')
        return self._RunMethod(config, request, global_params=global_params)
    GetCertificate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/keystores/{keystoresId}/aliases/{aliasesId}/certificate', http_method='GET', method_id='apigee.organizations.environments.keystores.aliases.getCertificate', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}/certificate', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsKeystoresAliasesGetCertificateRequest', response_type_name='GoogleApiHttpBody', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the certificate in an alias.

      Args:
        request: (ApigeeOrganizationsEnvironmentsKeystoresAliasesUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Alias) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/keystores/{keystoresId}/aliases/{aliasesId}', http_method='PUT', method_id='apigee.organizations.environments.keystores.aliases.update', ordered_params=['name'], path_params=['name'], query_params=['ignoreExpiryValidation', 'ignoreNewlineValidation'], relative_path='v1/{+name}', request_field='googleApiHttpBody', request_type_name='ApigeeOrganizationsEnvironmentsKeystoresAliasesUpdateRequest', response_type_name='GoogleCloudApigeeV1Alias', supports_download=False)