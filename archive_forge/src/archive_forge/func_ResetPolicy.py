from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.orgpolicy import service as org_policy_service
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.org_policies import arguments
from googlecloudsdk.command_lib.org_policies import exceptions
from googlecloudsdk.command_lib.org_policies import utils
from googlecloudsdk.core import log
def ResetPolicy(self, policy, update_mask):
    """Sets the reset field on the policy to True.

    If reset is set to True, no rules can be set on the policy and
    inheritFromParent has to be False. As such, this also deletes all rules on
    the policy and sets inheritFromParent to False.

    Args:
      policy: messages.GoogleCloudOrgpolicyV2alpha1Policy, The policy to be
        updated.
      update_mask: Specifies whether live/dryrun spec needs to be reset.

    Returns:
      The updated policy.
    """
    new_policy = copy.deepcopy(policy)
    if update_mask is None and new_policy.dryRunSpec:
        raise exceptions.InvalidInputError('update_mask is required if there is dry_run_spec in the request.')
    if self.ShouldUpdateLiveSpec(update_mask):
        if not new_policy.spec:
            new_policy.spec = self.org_policy_api.CreateEmptyPolicySpec()
        new_policy.spec.reset = True
        new_policy.spec.rules = []
        new_policy.spec.inheritFromParent = False
    if self.ShouldUpdateDryRunSpec(update_mask):
        if not new_policy.dryRunSpec:
            new_policy.dryRunSpec = self.org_policy_api.CreateEmptyPolicySpec()
        new_policy.dryRunSpec.reset = True
        new_policy.dryRunSpec.rules = []
        new_policy.dryRunSpec.inheritFromParent = False
    return new_policy