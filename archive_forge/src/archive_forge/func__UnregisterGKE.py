from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.container.fleet import agent_util
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import exclusivity_util
from googlecloudsdk.command_lib.container.fleet import kube_util
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet import util as hub_util
from googlecloudsdk.command_lib.container.fleet.memberships import gke_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def _UnregisterGKE(self, gke_cluster_resource_link, gke_cluster_uri, project, location, membership_id, args):
    """Register a GKE cluster without installing Connect agent."""
    try:
        name = 'projects/{}/locations/{}/memberships/{}'.format(project, location, membership_id)
        obj = api_util.GetMembership(name, self.ReleaseTrack())
        if obj.endpoint.gkeCluster.resourceLink != gke_cluster_resource_link:
            raise exceptions.Error('membership [{0}] is associated with a different GKE cluster link {1}. You may be unregistering the wrong membership.'.format(name, obj.endpoint.gkeCluster.resourceLink))
        api_util.DeleteMembership(name, self.ReleaseTrack())
    except apitools_exceptions.HttpUnauthorizedError as e:
        raise exceptions.Error('You are not authorized to unregister clusters from project [{}]. Underlying error: {}'.format(project, e))
    except apitools_exceptions.HttpNotFoundError:
        log.status.Print('Membership [{}] for the cluster was not found on the fleet. It may already have been deleted, or it may never have existed.'.format(name))