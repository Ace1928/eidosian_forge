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
def ListRBACRoleBindings(self, project, namespace):
    """Lists rolebindings in a namespace.

    Args:
      project: the project containing the namespace to list rolebindings from.
      namespace: the namespace to list rolebindings from.

    Returns:
      A ListNamespaceResponse (list of rolebindings and next page token)

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
    req = self.messages.GkehubProjectsLocationsNamespacesRbacrolebindingsListRequest(pageToken='', parent=util.RBACRoleBindingParentName(project, namespace))
    return list_pager.YieldFromList(self.client.projects_locations_namespaces_rbacrolebindings, req, field='rbacrolebindings', batch_size_attribute=None)