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
def GetRBACRoleBinding(self, name):
    """Gets an RBACRoleBinding resource from the GKEHub API.

    Args:
      name: the full rolebinding resource name.

    Returns:
      An RBACRoleBinding resource

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
    req = self.messages.GkehubProjectsLocationsNamespacesRbacrolebindingsGetRequest(name=name)
    return self.client.projects_locations_namespaces_rbacrolebindings.Get(req)