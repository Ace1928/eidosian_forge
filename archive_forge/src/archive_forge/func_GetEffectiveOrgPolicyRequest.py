from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resource_manager import exceptions
from googlecloudsdk.api_lib.resource_manager import org_policies
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import org_policies_base
from googlecloudsdk.command_lib.resource_manager import org_policies_flags as flags
@staticmethod
def GetEffectiveOrgPolicyRequest(args):
    m = org_policies.OrgPoliciesMessages()
    resource_id = org_policies_base.GetResource(args)
    request = m.GetEffectiveOrgPolicyRequest(constraint=org_policies.FormatConstraint(args.id))
    if args.project:
        return m.CloudresourcemanagerProjectsGetEffectiveOrgPolicyRequest(projectsId=resource_id, getEffectiveOrgPolicyRequest=request)
    elif args.organization:
        return m.CloudresourcemanagerOrganizationsGetEffectiveOrgPolicyRequest(organizationsId=resource_id, getEffectiveOrgPolicyRequest=request)
    elif args.folder:
        return m.CloudresourcemanagerFoldersGetEffectiveOrgPolicyRequest(foldersId=resource_id, getEffectiveOrgPolicyRequest=request)
    return None