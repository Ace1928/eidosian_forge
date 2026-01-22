from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.resource_manager import org_policies
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def _GetPolicy(project_id):
    """Get effective org policy of given project."""
    messages = org_policies.OrgPoliciesMessages()
    request = messages.CloudresourcemanagerProjectsGetEffectiveOrgPolicyRequest(projectsId=project_id, getEffectiveOrgPolicyRequest=messages.GetEffectiveOrgPolicyRequest(constraint=org_policies.FormatConstraint('compute.trustedImageProjects')))
    client = org_policies.OrgPoliciesClient()
    response = client.projects.GetEffectiveOrgPolicy(request)
    return response.listPolicy