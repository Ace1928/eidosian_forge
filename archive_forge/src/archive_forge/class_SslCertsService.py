from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
class SslCertsService(base_api.BaseApiService):
    """Service class for the sslCerts resource."""
    _NAME = 'sslCerts'

    def __init__(self, client):
        super(SqladminV1beta4.SslCertsService, self).__init__(client)
        self._upload_configs = {}

    def CreateEphemeral(self, request, global_params=None):
        """Generates a short-lived X509 certificate containing the provided public key and signed by a private key specific to the target instance. Users may use the certificate to authenticate as themselves when connecting to the database.

      Args:
        request: (SqlSslCertsCreateEphemeralRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SslCert) The response message.
      """
        config = self.GetMethodConfig('CreateEphemeral')
        return self._RunMethod(config, request, global_params=global_params)
    CreateEphemeral.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.sslCerts.createEphemeral', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/createEphemeral', request_field='sslCertsCreateEphemeralRequest', request_type_name='SqlSslCertsCreateEphemeralRequest', response_type_name='SslCert', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the SSL certificate. For First Generation instances, the certificate remains valid until the instance is restarted.

      Args:
        request: (SqlSslCertsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='sql.sslCerts.delete', ordered_params=['project', 'instance', 'sha1Fingerprint'], path_params=['instance', 'project', 'sha1Fingerprint'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/sslCerts/{sha1Fingerprint}', request_field='', request_type_name='SqlSslCertsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a particular SSL certificate. Does not include the private key (required for usage). The private key must be saved from the response to initial creation.

      Args:
        request: (SqlSslCertsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SslCert) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='sql.sslCerts.get', ordered_params=['project', 'instance', 'sha1Fingerprint'], path_params=['instance', 'project', 'sha1Fingerprint'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/sslCerts/{sha1Fingerprint}', request_field='', request_type_name='SqlSslCertsGetRequest', response_type_name='SslCert', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates an SSL certificate and returns it along with the private key and server certificate authority. The new certificate will not be usable until the instance is restarted.

      Args:
        request: (SqlSslCertsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SslCertsInsertResponse) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.sslCerts.insert', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/sslCerts', request_field='sslCertsInsertRequest', request_type_name='SqlSslCertsInsertRequest', response_type_name='SslCertsInsertResponse', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all of the current SSL certificates for the instance.

      Args:
        request: (SqlSslCertsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SslCertsListResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='sql.sslCerts.list', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/sslCerts', request_field='', request_type_name='SqlSslCertsListRequest', response_type_name='SslCertsListResponse', supports_download=False)