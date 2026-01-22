from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recaptchaenterprise.v1 import recaptchaenterprise_v1_messages as messages
class ProjectsRelatedaccountgroupmembershipsService(base_api.BaseApiService):
    """Service class for the projects_relatedaccountgroupmemberships resource."""
    _NAME = 'projects_relatedaccountgroupmemberships'

    def __init__(self, client):
        super(RecaptchaenterpriseV1.ProjectsRelatedaccountgroupmembershipsService, self).__init__(client)
        self._upload_configs = {}

    def Search(self, request, global_params=None):
        """Search group memberships related to a given account.

      Args:
        request: (RecaptchaenterpriseProjectsRelatedaccountgroupmembershipsSearchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1SearchRelatedAccountGroupMembershipsResponse) The response message.
      """
        config = self.GetMethodConfig('Search')
        return self._RunMethod(config, request, global_params=global_params)
    Search.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/relatedaccountgroupmemberships:search', http_method='POST', method_id='recaptchaenterprise.projects.relatedaccountgroupmemberships.search', ordered_params=['project'], path_params=['project'], query_params=[], relative_path='v1/{+project}/relatedaccountgroupmemberships:search', request_field='googleCloudRecaptchaenterpriseV1SearchRelatedAccountGroupMembershipsRequest', request_type_name='RecaptchaenterpriseProjectsRelatedaccountgroupmembershipsSearchRequest', response_type_name='GoogleCloudRecaptchaenterpriseV1SearchRelatedAccountGroupMembershipsResponse', supports_download=False)