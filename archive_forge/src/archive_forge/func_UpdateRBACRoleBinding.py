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
def UpdateRBACRoleBinding(self, name, user, group, role, mask):
    """Updates an RBACRoleBinding resource in the fleet.

    Args:
      name: the rolebinding name.
      user: the user.
      group: the group.
      role: the role.
      mask: a mask of the fields to update.

    Returns:
      An operation

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
    rolebinding = self.messages.RBACRoleBinding(name=name, user=user, group=group, role=self.messages.Role(predefinedRole=self.messages.Role.PredefinedRoleValueValuesEnum(role.upper())))
    req = self.messages.GkehubProjectsLocationsNamespacesRbacrolebindingsPatchRequest(rBACRoleBinding=rolebinding, name=rolebinding.name, updateMask=mask)
    return self.client.projects_locations_namespaces_rbacrolebindings.Patch(req)