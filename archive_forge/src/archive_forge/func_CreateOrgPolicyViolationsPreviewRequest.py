from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def CreateOrgPolicyViolationsPreviewRequest(self, violations_preview=None, parent=None):
    return self.messages.PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsCreateRequest(googleCloudPolicysimulatorV1OrgPolicyViolationsPreview=violations_preview, parent=parent)