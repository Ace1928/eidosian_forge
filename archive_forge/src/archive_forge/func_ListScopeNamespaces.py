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
def ListScopeNamespaces(self, parent):
    """Lists namespaces in a project.

    Args:
      parent: Full resource path of the scope containing the namespace.

    Returns:
      A ListNamespaceResponse (list of namespaces and next page token).

    Raises:
      apitools.base.py.HttpError: If the request returns an HTTP error.
    """
    req = self.messages.GkehubProjectsLocationsScopesNamespacesListRequest(pageToken='', parent=parent)
    return list_pager.YieldFromList(self.client.projects_locations_scopes_namespaces, req, field='scopeNamespaces', batch_size_attribute=None)