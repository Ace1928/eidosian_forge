from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.fleet import gkehub_api_util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.container.fleet import base as hub_base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
import six
def DeleteMembership(name, release_track=None):
    """Deletes a membership from the GKE Hub.

  Args:
    name: the full resource name of the membership to delete, e.g.,
      projects/foo/locations/global/memberships/name.
    release_track: the release_track used in the gcloud command, or None if it
      is not available.

  Raises:
    apitools.base.py.HttpError: if the request returns an HTTP error
  """
    client = gkehub_api_util.GetApiClientForTrack(release_track)
    op = client.projects_locations_memberships.Delete(client.MESSAGES_MODULE.GkehubProjectsLocationsMembershipsDeleteRequest(name=name))
    op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
    waiter.WaitFor(waiter.CloudOperationPollerNoResources(client.projects_locations_operations), op_resource, 'Waiting for membership to be deleted')