from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from typing import Generator
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as messages
def UpdateScopeRBACRoleBinding(self, name, user, group, role, labels, mask):
    """Updates an ScopeRBACRoleBinding resource in the fleet.

    Args:
      name: the rolebinding name.
      user: the user.
      group: the group.
      role: the role.
      labels: labels for the RBACRoleBinding resource.
      mask: a mask of the fields to update.

    Returns:
      An operation

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
    rolebinding = self.messages.RBACRoleBinding(name=name, user=user, group=group, role=self.messages.Role(predefinedRole=self.messages.Role.PredefinedRoleValueValuesEnum(role.upper())), labels=labels)
    req = self.messages.GkehubProjectsLocationsScopesRbacrolebindingsPatchRequest(rBACRoleBinding=rolebinding, name=rolebinding.name, updateMask=mask)
    op = self.client.projects_locations_scopes_rbacrolebindings.Patch(req)
    op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
    return waiter.WaitFor(waiter.CloudOperationPoller(self.client.projects_locations_scopes_rbacrolebindings, self.client.projects_locations_operations), op_resource, 'Waiting for rbacrolebinding to be updated')