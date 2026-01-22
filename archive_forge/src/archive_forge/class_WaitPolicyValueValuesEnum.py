from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WaitPolicyValueValuesEnum(_messages.Enum):
    """Optional. WaitForDeployPolicy delays a `Rollout` repair when a deploy
    policy violation is encountered.

    Values:
      WAIT_FOR_DEPLOY_POLICY_UNSPECIFIED: No WaitForDeployPolicy is specified.
      NEVER: Never waits on DeployPolicy, terminates `AutomationRun` if
        DeployPolicy check failed.
      LATEST: When policy passes, execute the latest `AutomationRun` only.
    """
    WAIT_FOR_DEPLOY_POLICY_UNSPECIFIED = 0
    NEVER = 1
    LATEST = 2