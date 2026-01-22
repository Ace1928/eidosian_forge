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
def DeleteScopeNamespace(self, namespace_path):
    """Deletes a namespace resource from the fleet.

    Args:
      namespace_path: Full resource path of the namespace.

    Returns:
      A long running operation for deleting the namespace.

    Raises:
      apitools.base.py.HttpError: If the request returns an HTTP error.
    """
    req = self.messages.GkehubProjectsLocationsScopesNamespacesDeleteRequest(name=namespace_path)
    op = self.client.projects_locations_scopes_namespaces.Delete(req)
    op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
    return waiter.WaitFor(waiter.CloudOperationPollerNoResources(self.client.projects_locations_operations), op_resource, 'Waiting for namespace to be deleted')