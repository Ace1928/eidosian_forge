from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resource_manager import org_policies
def OrgPoliciesService(args):
    client = org_policies.OrgPoliciesClient()
    if args.project:
        return client.projects
    elif args.organization:
        return client.organizations
    elif args.folder:
        return client.folders
    else:
        return None