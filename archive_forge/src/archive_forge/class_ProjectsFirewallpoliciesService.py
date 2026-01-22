from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recaptchaenterprise.v1 import recaptchaenterprise_v1_messages as messages
class ProjectsFirewallpoliciesService(base_api.BaseApiService):
    """Service class for the projects_firewallpolicies resource."""
    _NAME = 'projects_firewallpolicies'

    def __init__(self, client):
        super(RecaptchaenterpriseV1.ProjectsFirewallpoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new FirewallPolicy, specifying conditions at which reCAPTCHA Enterprise actions can be executed. A project may have a maximum of 1000 policies.

      Args:
        request: (RecaptchaenterpriseProjectsFirewallpoliciesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1FirewallPolicy) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/firewallpolicies', http_method='POST', method_id='recaptchaenterprise.projects.firewallpolicies.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/firewallpolicies', request_field='googleCloudRecaptchaenterpriseV1FirewallPolicy', request_type_name='RecaptchaenterpriseProjectsFirewallpoliciesCreateRequest', response_type_name='GoogleCloudRecaptchaenterpriseV1FirewallPolicy', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified firewall policy.

      Args:
        request: (RecaptchaenterpriseProjectsFirewallpoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/firewallpolicies/{firewallpoliciesId}', http_method='DELETE', method_id='recaptchaenterprise.projects.firewallpolicies.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='RecaptchaenterpriseProjectsFirewallpoliciesDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified firewall policy.

      Args:
        request: (RecaptchaenterpriseProjectsFirewallpoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1FirewallPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/firewallpolicies/{firewallpoliciesId}', http_method='GET', method_id='recaptchaenterprise.projects.firewallpolicies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='RecaptchaenterpriseProjectsFirewallpoliciesGetRequest', response_type_name='GoogleCloudRecaptchaenterpriseV1FirewallPolicy', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the list of all firewall policies that belong to a project.

      Args:
        request: (RecaptchaenterpriseProjectsFirewallpoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1ListFirewallPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/firewallpolicies', http_method='GET', method_id='recaptchaenterprise.projects.firewallpolicies.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/firewallpolicies', request_field='', request_type_name='RecaptchaenterpriseProjectsFirewallpoliciesListRequest', response_type_name='GoogleCloudRecaptchaenterpriseV1ListFirewallPoliciesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified firewall policy.

      Args:
        request: (RecaptchaenterpriseProjectsFirewallpoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1FirewallPolicy) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/firewallpolicies/{firewallpoliciesId}', http_method='PATCH', method_id='recaptchaenterprise.projects.firewallpolicies.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudRecaptchaenterpriseV1FirewallPolicy', request_type_name='RecaptchaenterpriseProjectsFirewallpoliciesPatchRequest', response_type_name='GoogleCloudRecaptchaenterpriseV1FirewallPolicy', supports_download=False)

    def Reorder(self, request, global_params=None):
        """Reorders all firewall policies.

      Args:
        request: (RecaptchaenterpriseProjectsFirewallpoliciesReorderRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1ReorderFirewallPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('Reorder')
        return self._RunMethod(config, request, global_params=global_params)
    Reorder.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/firewallpolicies:reorder', http_method='POST', method_id='recaptchaenterprise.projects.firewallpolicies.reorder', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/firewallpolicies:reorder', request_field='googleCloudRecaptchaenterpriseV1ReorderFirewallPoliciesRequest', request_type_name='RecaptchaenterpriseProjectsFirewallpoliciesReorderRequest', response_type_name='GoogleCloudRecaptchaenterpriseV1ReorderFirewallPoliciesResponse', supports_download=False)