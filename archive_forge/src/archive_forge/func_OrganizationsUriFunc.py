from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudresourcemanager import organizations
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
def OrganizationsUriFunc(resource):
    """Get the Organization URI for the given resource."""
    registry = resources.REGISTRY.Clone()
    registry.RegisterApiByName('cloudresourcemanager', 'v1')
    org_id = StripOrgPrefix(resource.name)
    org_ref = registry.Parse(None, params={'organizationsId': org_id}, collection='cloudresourcemanager.organizations')
    return org_ref.SelfLink()