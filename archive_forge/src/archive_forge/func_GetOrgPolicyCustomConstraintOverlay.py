from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetOrgPolicyCustomConstraintOverlay(self, custom_constraint=None, custom_constraint_parent=None):
    return self.messages.GoogleCloudPolicysimulatorV1OrgPolicyOverlayCustomConstraintOverlay(customConstraint=custom_constraint, customConstraintParent=custom_constraint_parent)