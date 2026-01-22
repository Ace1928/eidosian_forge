from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsApiproductsAttributesService(base_api.BaseApiService):
    """Service class for the organizations_apiproducts_attributes resource."""
    _NAME = 'organizations_apiproducts_attributes'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsApiproductsAttributesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes an API product attribute.

      Args:
        request: (ApigeeOrganizationsApiproductsAttributesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Attribute) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apiproducts/{apiproductsId}/attributes/{attributesId}', http_method='DELETE', method_id='apigee.organizations.apiproducts.attributes.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsApiproductsAttributesDeleteRequest', response_type_name='GoogleCloudApigeeV1Attribute', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the value of an API product attribute.

      Args:
        request: (ApigeeOrganizationsApiproductsAttributesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Attribute) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apiproducts/{apiproductsId}/attributes/{attributesId}', http_method='GET', method_id='apigee.organizations.apiproducts.attributes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsApiproductsAttributesGetRequest', response_type_name='GoogleCloudApigeeV1Attribute', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all API product attributes.

      Args:
        request: (ApigeeOrganizationsApiproductsAttributesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Attributes) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apiproducts/{apiproductsId}/attributes', http_method='GET', method_id='apigee.organizations.apiproducts.attributes.list', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/attributes', request_field='', request_type_name='ApigeeOrganizationsApiproductsAttributesListRequest', response_type_name='GoogleCloudApigeeV1Attributes', supports_download=False)

    def UpdateApiProductAttribute(self, request, global_params=None):
        """Updates the value of an API product attribute. **Note**: OAuth access tokens and Key Management Service (KMS) entities (apps, developers, and API products) are cached for 180 seconds (current default). Any custom attributes associated with entities also get cached for at least 180 seconds after entity is accessed during runtime. In this case, the `ExpiresIn` element on the OAuthV2 policy won't be able to expire an access token in less than 180 seconds.

      Args:
        request: (GoogleCloudApigeeV1Attribute) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Attribute) The response message.
      """
        config = self.GetMethodConfig('UpdateApiProductAttribute')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateApiProductAttribute.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apiproducts/{apiproductsId}/attributes/{attributesId}', http_method='POST', method_id='apigee.organizations.apiproducts.attributes.updateApiProductAttribute', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='<request>', request_type_name='GoogleCloudApigeeV1Attribute', response_type_name='GoogleCloudApigeeV1Attribute', supports_download=False)