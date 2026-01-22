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
def UpdateMembership(name, membership, update_mask, release_track, description=None, external_id=None, infra_type=None, clear_labels=False, update_labels=None, remove_labels=None, issuer_url=None, oidc_jwks=None, api_server_version=None, async_flag=False):
    """UpdateMembership updates membership resource in the GKE Hub API.

  Args:
    name: The full resource name of the membership to update, e.g.
      projects/foo/locations/global/memberships/name.
    membership: Membership resource that needs to be updated.
    update_mask: Field names of membership resource to be updated.
    release_track: The release_track used in the gcloud command.
    description: the value to put in the description field
    external_id: the unique id associated with the cluster, or None if it is not
      available.
    infra_type: The infrastructure type that the cluster is running on
    clear_labels: Whether labels should be cleared
    update_labels: Labels to be updated,
    remove_labels: Labels to be removed,
    issuer_url: The discovery URL for the cluster's service account token
      issuer.
    oidc_jwks: The JSON Web Key Set string containing public keys for validating
      service account tokens. Set to None if the issuer_url is
      publicly-reachable. Still requires issuer_url to be set.
    api_server_version: api_server_version of the cluster
    async_flag: Whether to return the update operation instead of polling

  Returns:
    The updated Membership resource or the update operation if async.

  Raises:
    - apitools.base.py.HttpError: if the request returns an HTTP error
    - exceptions raised by waiter.WaitFor()
  """
    client = gkehub_api_util.GetApiClientForTrack(release_track)
    messages = client.MESSAGES_MODULE
    request = messages.GkehubProjectsLocationsMembershipsPatchRequest(membership=membership, name=name, updateMask=update_mask)
    if issuer_url:
        request.membership.authority = messages.Authority(issuer=issuer_url)
        if oidc_jwks:
            request.membership.authority.oidcJwks = oidc_jwks.encode('utf-8')
        else:
            request.membership.authority.oidcJwks = None
    else:
        request.membership.authority = None
    if api_server_version:
        resource_options = messages.ResourceOptions(k8sVersion=api_server_version)
        kubernetes_resource = messages.KubernetesResource(resourceOptions=resource_options)
        endpoint = messages.MembershipEndpoint(kubernetesResource=kubernetes_resource)
        if request.membership.endpoint:
            if request.membership.endpoint.kubernetesResource:
                if request.membership.endpoint.kubernetesResource.resourceOptions:
                    request.membership.endpoint.kubernetesResource.resourceOptions.k8sVersion = api_server_version
                else:
                    request.membership.endpoint.kubernetesResource.resourceOptions = resource_options
            else:
                request.membership.endpoint.kubernetesResource = kubernetes_resource
        else:
            request.membership.endpoint = endpoint
    if description:
        request.membership.description = description
    if external_id:
        request.membership.externalId = external_id
    if infra_type == 'on-prem':
        request.membership.infrastructureType = messages.Membership.InfrastructureTypeValueValuesEnum.ON_PREM
    elif infra_type == 'multi-cloud':
        request.membership.infrastructureType = messages.Membership.InfrastructureTypeValueValuesEnum.MULTI_CLOUD
    if clear_labels or update_labels or remove_labels:
        mem_labels = {}
        if not clear_labels and membership.labels:
            for item in membership.labels.additionalProperties:
                mem_labels[item.key] = six.text_type(item.value)
        if update_labels:
            for k, v in sorted(six.iteritems(update_labels)):
                mem_labels[k] = v
        if remove_labels:
            for k in remove_labels:
                if k in mem_labels:
                    mem_labels.pop(k)
        labels = messages.Membership.LabelsValue()
        for k, v in sorted(six.iteritems(mem_labels)):
            labels.additionalProperties.append(labels.AdditionalProperty(key=k, value=v))
        request.membership.labels = labels
    op = client.projects_locations_memberships.Patch(request)
    log.status.Print('request issued for: [{}]'.format(name))
    if async_flag:
        log.status.Print('Check operation [{}] for status.'.format(op.name))
        return op
    op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
    return waiter.WaitFor(waiter.CloudOperationPoller(client.projects_locations_memberships, client.projects_locations_operations), op_resource, 'Waiting for operation [{}] to complete'.format(op.name))