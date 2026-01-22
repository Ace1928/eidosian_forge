from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resource_manager import org_policies
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import org_policies_base
from googlecloudsdk.command_lib.resource_manager import org_policies_flags as flags
from googlecloudsdk.core import log
@staticmethod
def ClearOrgPolicyRequest(args):
    messages = org_policies.OrgPoliciesMessages()
    resource_id = org_policies_base.GetResource(args)
    request = messages.ClearOrgPolicyRequest(constraint=org_policies.FormatConstraint(args.id))
    if args.project:
        return messages.CloudresourcemanagerProjectsClearOrgPolicyRequest(projectsId=resource_id, clearOrgPolicyRequest=request)
    elif args.organization:
        return messages.CloudresourcemanagerOrganizationsClearOrgPolicyRequest(organizationsId=resource_id, clearOrgPolicyRequest=request)
    elif args.folder:
        return messages.CloudresourcemanagerFoldersClearOrgPolicyRequest(foldersId=resource_id, clearOrgPolicyRequest=request)
    return None