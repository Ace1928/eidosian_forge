from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsInstancesNatAddressesService(base_api.BaseApiService):
    """Service class for the organizations_instances_natAddresses resource."""
    _NAME = 'organizations_instances_natAddresses'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsInstancesNatAddressesService, self).__init__(client)
        self._upload_configs = {}

    def Activate(self, request, global_params=None):
        """Activates the NAT address. The Apigee instance can now use this for Internet egress traffic. **Note:** Not supported for Apigee hybrid.

      Args:
        request: (ApigeeOrganizationsInstancesNatAddressesActivateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Activate')
        return self._RunMethod(config, request, global_params=global_params)
    Activate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/instances/{instancesId}/natAddresses/{natAddressesId}:activate', http_method='POST', method_id='apigee.organizations.instances.natAddresses.activate', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:activate', request_field='googleCloudApigeeV1ActivateNatAddressRequest', request_type_name='ApigeeOrganizationsInstancesNatAddressesActivateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a NAT address. The address is created in the RESERVED state and a static external IP address will be provisioned. At this time, the instance will not use this IP address for Internet egress traffic. The address can be activated for use once any required firewall IP whitelisting has been completed. **Note:** Not supported for Apigee hybrid.

      Args:
        request: (ApigeeOrganizationsInstancesNatAddressesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/instances/{instancesId}/natAddresses', http_method='POST', method_id='apigee.organizations.instances.natAddresses.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/natAddresses', request_field='googleCloudApigeeV1NatAddress', request_type_name='ApigeeOrganizationsInstancesNatAddressesCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the NAT address. Connections that are actively using the address are drained before it is removed. **Note:** Not supported for Apigee hybrid.

      Args:
        request: (ApigeeOrganizationsInstancesNatAddressesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/instances/{instancesId}/natAddresses/{natAddressesId}', http_method='DELETE', method_id='apigee.organizations.instances.natAddresses.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsInstancesNatAddressesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the details of a NAT address. **Note:** Not supported for Apigee hybrid.

      Args:
        request: (ApigeeOrganizationsInstancesNatAddressesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1NatAddress) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/instances/{instancesId}/natAddresses/{natAddressesId}', http_method='GET', method_id='apigee.organizations.instances.natAddresses.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsInstancesNatAddressesGetRequest', response_type_name='GoogleCloudApigeeV1NatAddress', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the NAT addresses for an Apigee instance. **Note:** Not supported for Apigee hybrid.

      Args:
        request: (ApigeeOrganizationsInstancesNatAddressesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListNatAddressesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/instances/{instancesId}/natAddresses', http_method='GET', method_id='apigee.organizations.instances.natAddresses.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/natAddresses', request_field='', request_type_name='ApigeeOrganizationsInstancesNatAddressesListRequest', response_type_name='GoogleCloudApigeeV1ListNatAddressesResponse', supports_download=False)