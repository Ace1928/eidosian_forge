import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib import network_services
from googlecloudsdk.api_lib.container import util as container_util
from googlecloudsdk.api_lib.container.fleet import util as fleet_util
from googlecloudsdk.command_lib.container.fleet import api_util as hubapi_util
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
def ContextGenerator(args):
    """Generate k8s context from membership, location and project."""
    membership_resource_name = base.ParseMembership(args, prompt=True, autoselect=True, search=True)
    membership_id = fleet_util.MembershipShortname(membership_resource_name)
    project_id = args.project
    location = args.location
    if project_id is None:
        project_id = properties.VALUES.core.project.Get()
    try:
        membership_resource = hubapi_util.GetMembership(membership_resource_name)
    except apitools_exceptions.HttpNotFoundError:
        raise exceptions.Error('Failed finding membership. Please verify the membership, and location passed is valid.  membership={}, location={}, project={}'.format(membership_id, location, project_id))
    if membership_resource is None:
        print('Membership resource is none')
        return
    if not membership_resource.endpoint.gkeCluster:
        raise exceptions.Error('The cluster to the input membership does not belong to gke. Please verify the membership and location passed is valid.  membership={}, location={}, project={}.'.format(membership_id, location, project_id))
    cluster_resourcelink = membership_resource.endpoint.gkeCluster.resourceLink
    cluster_location = cluster_resourcelink.split('/')[-3]
    cluster_name = cluster_resourcelink.split('/')[-1]
    print('Found cluster=' + cluster_name)
    cluster_context = container_util.ClusterConfig.KubeContext(cluster_name, cluster_location, project_id)
    return cluster_context