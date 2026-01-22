from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def NFSNetworkFullName(nfs_share_resource, allowed_client_dict):
    """Returns the full GCP name of the NFS allowed client network."""
    nfs_region = nfs_share_resource.Parent()
    nfs_project = nfs_region.Parent()
    network_project_id = allowed_client_dict.get('network-project-id', nfs_project.Name())
    return resources.REGISTRY.Parse(allowed_client_dict['network'], params={'projectsId': network_project_id, 'locationsId': nfs_region.Name()}, collection='baremetalsolution.projects.locations.networks').RelativeName()