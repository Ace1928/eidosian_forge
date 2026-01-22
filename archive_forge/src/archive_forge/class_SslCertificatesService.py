from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class SslCertificatesService(base_api.BaseApiService):
    """Service class for the sslCertificates resource."""
    _NAME = 'sslCertificates'

    def __init__(self, client):
        super(ComputeBeta.SslCertificatesService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves the list of all SslCertificate resources, regional and global, available to the specified project. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeSslCertificatesAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SslCertificateAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.sslCertificates.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/sslCertificates', request_field='', request_type_name='ComputeSslCertificatesAggregatedListRequest', response_type_name='SslCertificateAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified SslCertificate resource.

      Args:
        request: (ComputeSslCertificatesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.sslCertificates.delete', ordered_params=['project', 'sslCertificate'], path_params=['project', 'sslCertificate'], query_params=['requestId'], relative_path='projects/{project}/global/sslCertificates/{sslCertificate}', request_field='', request_type_name='ComputeSslCertificatesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified SslCertificate resource.

      Args:
        request: (ComputeSslCertificatesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SslCertificate) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.sslCertificates.get', ordered_params=['project', 'sslCertificate'], path_params=['project', 'sslCertificate'], query_params=[], relative_path='projects/{project}/global/sslCertificates/{sslCertificate}', request_field='', request_type_name='ComputeSslCertificatesGetRequest', response_type_name='SslCertificate', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a SslCertificate resource in the specified project using the data included in the request.

      Args:
        request: (ComputeSslCertificatesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.sslCertificates.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/sslCertificates', request_field='sslCertificate', request_type_name='ComputeSslCertificatesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of SslCertificate resources available to the specified project.

      Args:
        request: (ComputeSslCertificatesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SslCertificateList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.sslCertificates.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/sslCertificates', request_field='', request_type_name='ComputeSslCertificatesListRequest', response_type_name='SslCertificateList', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeSslCertificatesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.sslCertificates.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/sslCertificates/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeSslCertificatesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)