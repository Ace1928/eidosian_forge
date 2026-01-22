from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkehub.v1beta import gkehub_v1beta_messages as messages
def GenerateMembershipRBACRoleBindingYAML(self, request, global_params=None):
    """Generates a YAML of the RBAC policies for the specified RoleBinding and its associated impersonation resources.

      Args:
        request: (GkehubProjectsLocationsMembershipsRbacrolebindingsGenerateMembershipRBACRoleBindingYAMLRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GenerateMembershipRBACRoleBindingYAMLResponse) The response message.
      """
    config = self.GetMethodConfig('GenerateMembershipRBACRoleBindingYAML')
    return self._RunMethod(config, request, global_params=global_params)