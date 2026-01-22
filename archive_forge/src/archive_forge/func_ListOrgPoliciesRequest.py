from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resource_manager import org_policies
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import org_policies_base
from googlecloudsdk.command_lib.resource_manager import org_policies_flags as flags
@staticmethod
def ListOrgPoliciesRequest(args):
    messages = org_policies.OrgPoliciesMessages()
    resource_id = org_policies_base.GetResource(args)
    request = messages.ListOrgPoliciesRequest()
    if args.project:
        return messages.CloudresourcemanagerProjectsListOrgPoliciesRequest(projectsId=resource_id, listOrgPoliciesRequest=request)
    elif args.organization:
        return messages.CloudresourcemanagerOrganizationsListOrgPoliciesRequest(organizationsId=resource_id, listOrgPoliciesRequest=request)
    elif args.folder:
        return messages.CloudresourcemanagerFoldersListOrgPoliciesRequest(foldersId=resource_id, listOrgPoliciesRequest=request)
    return None