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
def ProjectForClusterUUID(uuid, projects, release_track=None):
    """Retrieves the project that the cluster UUID has a Membership with.

  Args:
    uuid: the UUID of the cluster.
    projects: sequence of project IDs to consider.
    release_track: the release_track used in the gcloud command, or None if it
      is not available.

  Returns:
    a project ID.

  Raises:
    apitools.base.py.HttpError: if any request returns an HTTP error
  """
    client = gkehub_api_util.GetApiClientForTrack(release_track)
    for project in projects:
        if project:
            parent = 'projects/{}/locations/global'.format(project)
            membership_response = client.projects_locations_memberships.List(client.MESSAGES_MODULE.GkehubProjectsLocationsMembershipsListRequest(parent=parent))
            for membership in membership_response.resources:
                membership_uuid = _ClusterUUIDForMembershipName(membership.name)
                if membership_uuid == uuid:
                    return project
    return None