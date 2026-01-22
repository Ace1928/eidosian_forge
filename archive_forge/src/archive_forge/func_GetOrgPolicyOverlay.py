from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetOrgPolicyOverlay(self, custom_constraints=None, policies=None):
    return self.messages.GoogleCloudPolicysimulatorV1OrgPolicyOverlay(customConstraints=custom_constraints, policies=policies)