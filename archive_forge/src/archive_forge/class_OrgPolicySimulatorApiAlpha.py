from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class OrgPolicySimulatorApiAlpha(OrgPolicySimulatorApi):
    """Base Class for OrgPolicy Simulator API Alpha."""

    def CreateOrgPolicyViolationsPreviewRequest(self, violations_preview=None, parent=None):
        return self.messages.PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsCreateRequest(googleCloudPolicysimulatorV1alphaOrgPolicyViolationsPreview=violations_preview, parent=parent)

    def GetPolicysimulatorOrgPolicyViolationsPreview(self, name=None, overlay=None, resource_counts=None, state=None, violations_count=None):
        return self.messages.GoogleCloudPolicysimulatorV1alphaOrgPolicyViolationsPreview(name=name, overlay=overlay, resourceCounts=resource_counts, state=state, violationsCount=violations_count)

    def GetOrgPolicyOverlay(self, custom_constraints=None, policies=None):
        return self.messages.GoogleCloudPolicysimulatorV1alphaOrgPolicyOverlay(customConstraints=custom_constraints, policies=policies)

    def GetOrgPolicyPolicyOverlay(self, policy=None, policy_parent=None):
        return self.messages.GoogleCloudPolicysimulatorV1alphaOrgPolicyOverlayPolicyOverlay(policy=policy, policyParent=policy_parent)

    def GetOrgPolicyCustomConstraintOverlay(self, custom_constraint=None, custom_constraint_parent=None):
        return self.messages.GoogleCloudPolicysimulatorV1alphaOrgPolicyOverlayCustomConstraintOverlay(customConstraint=custom_constraint, customConstraintParent=custom_constraint_parent)

    def GetOrgPolicyViolationsPreviewMessage(self):
        m = self.messages.GoogleCloudPolicysimulatorV1alphaOrgPolicyViolationsPreview
        return m