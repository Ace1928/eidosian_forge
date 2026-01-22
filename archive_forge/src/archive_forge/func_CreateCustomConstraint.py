from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.orgpolicy import utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.generated_clients.apis.orgpolicy.v2 import orgpolicy_v2_messages
def CreateCustomConstraint(self, custom_constraint):
    parent = utils.GetResourceFromPolicyName(custom_constraint.name)
    request = self.messages.OrgpolicyOrganizationsCustomConstraintsCreateRequest(parent=parent, googleCloudOrgpolicyV2CustomConstraint=custom_constraint)
    return self.client.organizations_customConstraints.Create(request=request)