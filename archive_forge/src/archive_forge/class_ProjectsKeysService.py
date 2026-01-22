from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recaptchaenterprise.v1 import recaptchaenterprise_v1_messages as messages
class ProjectsKeysService(base_api.BaseApiService):
    """Service class for the projects_keys resource."""
    _NAME = 'projects_keys'

    def __init__(self, client):
        super(RecaptchaenterpriseV1.ProjectsKeysService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new reCAPTCHA Enterprise key.

      Args:
        request: (RecaptchaenterpriseProjectsKeysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1Key) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/keys', http_method='POST', method_id='recaptchaenterprise.projects.keys.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/keys', request_field='googleCloudRecaptchaenterpriseV1Key', request_type_name='RecaptchaenterpriseProjectsKeysCreateRequest', response_type_name='GoogleCloudRecaptchaenterpriseV1Key', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified key.

      Args:
        request: (RecaptchaenterpriseProjectsKeysDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/keys/{keysId}', http_method='DELETE', method_id='recaptchaenterprise.projects.keys.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='RecaptchaenterpriseProjectsKeysDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified key.

      Args:
        request: (RecaptchaenterpriseProjectsKeysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1Key) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/keys/{keysId}', http_method='GET', method_id='recaptchaenterprise.projects.keys.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='RecaptchaenterpriseProjectsKeysGetRequest', response_type_name='GoogleCloudRecaptchaenterpriseV1Key', supports_download=False)

    def GetMetrics(self, request, global_params=None):
        """Get some aggregated metrics for a Key. This data can be used to build dashboards.

      Args:
        request: (RecaptchaenterpriseProjectsKeysGetMetricsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1Metrics) The response message.
      """
        config = self.GetMethodConfig('GetMetrics')
        return self._RunMethod(config, request, global_params=global_params)
    GetMetrics.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/keys/{keysId}/metrics', http_method='GET', method_id='recaptchaenterprise.projects.keys.getMetrics', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='RecaptchaenterpriseProjectsKeysGetMetricsRequest', response_type_name='GoogleCloudRecaptchaenterpriseV1Metrics', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the list of all keys that belong to a project.

      Args:
        request: (RecaptchaenterpriseProjectsKeysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1ListKeysResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/keys', http_method='GET', method_id='recaptchaenterprise.projects.keys.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/keys', request_field='', request_type_name='RecaptchaenterpriseProjectsKeysListRequest', response_type_name='GoogleCloudRecaptchaenterpriseV1ListKeysResponse', supports_download=False)

    def Migrate(self, request, global_params=None):
        """Migrates an existing key from reCAPTCHA to reCAPTCHA Enterprise. Once a key is migrated, it can be used from either product. SiteVerify requests are billed as CreateAssessment calls. You must be authenticated as one of the current owners of the reCAPTCHA Key, and your user must have the reCAPTCHA Enterprise Admin IAM role in the destination project.

      Args:
        request: (RecaptchaenterpriseProjectsKeysMigrateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1Key) The response message.
      """
        config = self.GetMethodConfig('Migrate')
        return self._RunMethod(config, request, global_params=global_params)
    Migrate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/keys/{keysId}:migrate', http_method='POST', method_id='recaptchaenterprise.projects.keys.migrate', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:migrate', request_field='googleCloudRecaptchaenterpriseV1MigrateKeyRequest', request_type_name='RecaptchaenterpriseProjectsKeysMigrateRequest', response_type_name='GoogleCloudRecaptchaenterpriseV1Key', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified key.

      Args:
        request: (RecaptchaenterpriseProjectsKeysPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1Key) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/keys/{keysId}', http_method='PATCH', method_id='recaptchaenterprise.projects.keys.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudRecaptchaenterpriseV1Key', request_type_name='RecaptchaenterpriseProjectsKeysPatchRequest', response_type_name='GoogleCloudRecaptchaenterpriseV1Key', supports_download=False)

    def RetrieveLegacySecretKey(self, request, global_params=None):
        """Returns the secret key related to the specified public key. You must use the legacy secret key only in a 3rd party integration with legacy reCAPTCHA.

      Args:
        request: (RecaptchaenterpriseProjectsKeysRetrieveLegacySecretKeyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1RetrieveLegacySecretKeyResponse) The response message.
      """
        config = self.GetMethodConfig('RetrieveLegacySecretKey')
        return self._RunMethod(config, request, global_params=global_params)
    RetrieveLegacySecretKey.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/keys/{keysId}:retrieveLegacySecretKey', http_method='GET', method_id='recaptchaenterprise.projects.keys.retrieveLegacySecretKey', ordered_params=['key'], path_params=['key'], query_params=[], relative_path='v1/{+key}:retrieveLegacySecretKey', request_field='', request_type_name='RecaptchaenterpriseProjectsKeysRetrieveLegacySecretKeyRequest', response_type_name='GoogleCloudRecaptchaenterpriseV1RetrieveLegacySecretKeyResponse', supports_download=False)