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
def UpdateNamespace(self, name, scope, project, mask):
    """Updates a namespace resource in the fleet.

    Args:
      name: the namespace name.
      scope: the scope containing the namespace.
      project: the project containing the namespace.
      mask: a mask of the fields to update.

    Returns:
      An operation

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
    namespace = self.messages.Namespace(name=util.NamespaceResourceName(project, name), scope=scope)
    req = self.messages.GkehubProjectsLocationsNamespacesPatchRequest(namespace=namespace, name=util.NamespaceResourceName(project, name), updateMask=mask)
    op = self.client.projects_locations_namespaces.Patch(req)
    op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
    return waiter.WaitFor(waiter.CloudOperationPoller(self.client.projects_locations_namespaces, self.client.projects_locations_operations), op_resource, 'Waiting for namespace to be updated')