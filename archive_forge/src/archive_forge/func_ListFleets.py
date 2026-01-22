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
def ListFleets(self, project, organization):
    """Lists fleets in an organization.

    Args:
      project: the project to search.
      organization: the organization to search.

    Returns:
      A ListFleetResponse (list of fleets and next page token)

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
    if organization:
        parent = util.FleetOrgParentName(organization)
    else:
        parent = util.FleetParentName(project)
    req = self.messages.GkehubProjectsLocationsFleetsListRequest(pageToken='', parent=parent)
    return list_pager.YieldFromList(self.client.projects_locations_fleets, req, field='fleets', batch_size_attribute=None)