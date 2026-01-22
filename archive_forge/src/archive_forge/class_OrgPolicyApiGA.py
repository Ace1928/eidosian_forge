from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.orgpolicy import utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.generated_clients.apis.orgpolicy.v2 import orgpolicy_v2_messages
class OrgPolicyApiGA(OrgPolicyApi):
    """Base class for all Org Policy V2GA API."""

    def GetPolicy(self, name):
        if name.startswith('organizations/'):
            request = self.messages.OrgpolicyOrganizationsPoliciesGetRequest(name=name)
            return self.client.organizations_policies.Get(request)
        elif name.startswith('folders/'):
            request = self.messages.OrgpolicyFoldersPoliciesGetRequest(name=name)
            return self.client.folders_policies.Get(request)
        else:
            request = self.messages.OrgpolicyProjectsPoliciesGetRequest(name=name)
            return self.client.projects_policies.Get(request)

    def GetEffectivePolicy(self, name):
        if name.startswith('organizations/'):
            request = self.messages.OrgpolicyOrganizationsPoliciesGetEffectivePolicyRequest(name=name)
            return self.client.organizations_policies.GetEffectivePolicy(request)
        elif name.startswith('folders/'):
            request = self.messages.OrgpolicyFoldersPoliciesGetEffectivePolicyRequest(name=name)
            return self.client.folders_policies.GetEffectivePolicy(request)
        else:
            request = self.messages.OrgpolicyProjectsPoliciesGetEffectivePolicyRequest(name=name)
            return self.client.projects_policies.GetEffectivePolicy(request)

    def DeletePolicy(self, name: str, etag=None) -> orgpolicy_v2_messages.GoogleProtobufEmpty:
        if name.startswith('organizations/'):
            request = self.messages.OrgpolicyOrganizationsPoliciesDeleteRequest(name=name, etag=etag)
            return self.client.organizations_policies.Delete(request)
        elif name.startswith('folders/'):
            request = self.messages.OrgpolicyFoldersPoliciesDeleteRequest(name=name, etag=etag)
            return self.client.folders_policies.Delete(request)
        else:
            request = self.messages.OrgpolicyProjectsPoliciesDeleteRequest(name=name, etag=etag)
            return self.client.projects_policies.Delete(request)

    def ListPolicies(self, parent):
        if parent.startswith('organizations/'):
            request = self.messages.OrgpolicyOrganizationsPoliciesListRequest(parent=parent)
            return self.client.organizations_policies.List(request)
        elif parent.startswith('folders/'):
            request = self.messages.OrgpolicyFoldersPoliciesListRequest(parent=parent)
            return self.client.folders_policies.List(request)
        else:
            request = self.messages.OrgpolicyProjectsPoliciesListRequest(parent=parent)
            return self.client.projects_policies.List(request)

    def ListConstraints(self, parent):
        if parent.startswith('organizations/'):
            request = self.messages.OrgpolicyOrganizationsConstraintsListRequest(parent=parent)
            return self.client.organizations_constraints.List(request)
        elif parent.startswith('folders/'):
            request = self.messages.OrgpolicyFoldersConstraintsListRequest(parent=parent)
            return self.client.folders_constraints.List(request)
        else:
            request = self.messages.OrgpolicyProjectsConstraintsListRequest(parent=parent)
            return self.client.projects_constraints.List(request)

    def CreatePolicy(self, policy):
        parent = utils.GetResourceFromPolicyName(policy.name)
        if parent.startswith('organizations/'):
            request = self.messages.OrgpolicyOrganizationsPoliciesCreateRequest(parent=parent, googleCloudOrgpolicyV2Policy=policy)
            return self.client.organizations_policies.Create(request=request)
        elif parent.startswith('folders/'):
            request = self.messages.OrgpolicyFoldersPoliciesCreateRequest(parent=parent, googleCloudOrgpolicyV2Policy=policy)
            return self.client.folders_policies.Create(request=request)
        else:
            request = self.messages.OrgpolicyProjectsPoliciesCreateRequest(parent=parent, googleCloudOrgpolicyV2Policy=policy)
            return self.client.projects_policies.Create(request=request)

    def UpdatePolicy(self, policy, update_mask=None):
        if policy.name.startswith('organizations/'):
            request = self.messages.OrgpolicyOrganizationsPoliciesPatchRequest(name=policy.name, googleCloudOrgpolicyV2Policy=policy, updateMask=update_mask)
            return self.client.organizations_policies.Patch(request)
        elif policy.name.startswith('folders/'):
            request = self.messages.OrgpolicyFoldersPoliciesPatchRequest(name=policy.name, googleCloudOrgpolicyV2Policy=policy, updateMask=update_mask)
            return self.client.folders_policies.Patch(request)
        else:
            request = self.messages.OrgpolicyProjectsPoliciesPatchRequest(name=policy.name, googleCloudOrgpolicyV2Policy=policy, updateMask=update_mask)
            return self.client.projects_policies.Patch(request)

    def CreateCustomConstraint(self, custom_constraint):
        parent = utils.GetResourceFromPolicyName(custom_constraint.name)
        request = self.messages.OrgpolicyOrganizationsCustomConstraintsCreateRequest(parent=parent, googleCloudOrgpolicyV2CustomConstraint=custom_constraint)
        return self.client.organizations_customConstraints.Create(request=request)

    def UpdateCustomConstraint(self, custom_constraint):
        request = self.messages.OrgpolicyOrganizationsCustomConstraintsPatchRequest(googleCloudOrgpolicyV2CustomConstraint=custom_constraint, name=custom_constraint.name)
        return self.client.organizations_customConstraints.Patch(request)

    def GetCustomConstraint(self, name):
        request = self.messages.OrgpolicyOrganizationsCustomConstraintsGetRequest(name=name)
        return self.client.organizations_customConstraints.Get(request)

    def DeleteCustomConstraint(self, name):
        request = self.messages.OrgpolicyOrganizationsCustomConstraintsDeleteRequest(name=name)
        return self.client.organizations_customConstraints.Delete(request)

    def CreateEmptyPolicySpec(self):
        return self.messages.GoogleCloudOrgpolicyV2PolicySpec()

    def BuildPolicy(self, name):
        spec = self.messages.GoogleCloudOrgpolicyV2PolicySpec()
        return self.messages.GoogleCloudOrgpolicyV2Policy(name=name, spec=spec)

    def BuildEmptyPolicy(self, name, has_spec=False, has_dry_run_spec=False):
        spec = None
        dry_run_spec = None
        if has_spec:
            spec = self.messages.GoogleCloudOrgpolicyV2PolicySpec()
        if has_dry_run_spec:
            dry_run_spec = self.messages.GoogleCloudOrgpolicyV2PolicySpec()
        return self.messages.GoogleCloudOrgpolicyV2Policy(name=name, spec=spec, dryRunSpec=dry_run_spec)

    def BuildPolicySpecPolicyRule(self, condition=None, allow_all=None, deny_all=None, enforce=None, values=None):
        return self.messages.GoogleCloudOrgpolicyV2PolicySpecPolicyRule(condition=condition, allowAll=allow_all, denyAll=deny_all, enforce=enforce, values=values)

    def BuildPolicySpecPolicyRuleStringValues(self, allowed_values=(), denied_values=()):
        return self.messages.GoogleCloudOrgpolicyV2PolicySpecPolicyRuleStringValues(allowedValues=allowed_values, deniedValues=denied_values)