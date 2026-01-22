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
def ListPermittedScopes(self, project):
    """Lists scopes in a project permitted to be viewed by the caller.

    Args:
      project: Project containing the scope.

    Returns:
      A ListPermittedScopesResponse (list of permitted scopes and next page
      token).

    Raises:
      apitools.base.py.HttpError: If the request returns an HTTP error
    """
    parent = util.ScopeParentName(project)
    req = self.messages.GkehubProjectsLocationsScopesListPermittedRequest(pageToken='', parent=parent)
    return list_pager.YieldFromList(self.client.projects_locations_scopes, req, method='ListPermitted', field='scopes', batch_size_attribute=None)