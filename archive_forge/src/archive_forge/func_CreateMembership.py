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
def CreateMembership(project, membership_id, description, location=None, gke_cluster_self_link=None, external_id=None, release_track=None, issuer_url=None, oidc_jwks=None, api_server_version=None):
    """Creates a Membership resource in the GKE Hub API.

  Args:
    project: the project in which to create the membership
    membership_id: the value to use for the membership_id
    description: the value to put in the description field
    location: the location for the membership
    gke_cluster_self_link: the selfLink for the cluster if it is a GKE cluster,
      or None if it is not
    external_id: the unique id associated with the cluster, or None if it is not
      available.
    release_track: the release_track used in the gcloud command, or None if it
      is not available.
    issuer_url: the discovery URL for the cluster's service account token
      issuer. Set to None to skip enabling Workload Identity.
    oidc_jwks: the JSON Web Key Set containing public keys for validating
      service account tokens. Set to None if the issuer_url is
      publicly-routable. Still requires issuer_url to be set.
    api_server_version: api server version of the cluster for CRD

  Returns:
    the created Membership resource.

  Raises:
    - apitools.base.py.HttpError: if the request returns an HTTP error
    - exceptions raised by waiter.WaitFor()
  """
    client = gkehub_api_util.GetApiClientForTrack(release_track)
    messages = client.MESSAGES_MODULE
    parent_ref = ParentRef(project, location)
    request = messages.GkehubProjectsLocationsMembershipsCreateRequest(membership=messages.Membership(description=description), parent=parent_ref, membershipId=membership_id)
    if gke_cluster_self_link:
        endpoint = messages.MembershipEndpoint(gkeCluster=messages.GkeCluster(resourceLink=gke_cluster_self_link))
        request.membership.endpoint = endpoint
    elif api_server_version:
        resource_options = messages.ResourceOptions(k8sVersion=api_server_version)
        kubernetes_resource = messages.KubernetesResource(resourceOptions=resource_options)
        endpoint = messages.MembershipEndpoint(kubernetesResource=kubernetes_resource)
        request.membership.endpoint = endpoint
    if external_id:
        request.membership.externalId = external_id
    if issuer_url:
        request.membership.authority = messages.Authority(issuer=issuer_url)
        if oidc_jwks:
            request.membership.authority.oidcJwks = oidc_jwks.encode('utf-8')
    op = client.projects_locations_memberships.Create(request)
    op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
    return waiter.WaitFor(waiter.CloudOperationPoller(client.projects_locations_memberships, client.projects_locations_operations), op_resource, 'Waiting for membership to be created')