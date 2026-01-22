from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpgradeInfo(_messages.Message):
    """Information related to: * A function's eligibility for 1st Gen to 2nd
  Gen migration * Current state of migration for function undergoing
  migration.

  Enums:
    UpgradeStateValueValuesEnum: UpgradeState of the function

  Fields:
    buildConfig: Describes the Build step of the function that builds a
      container to prepare for 2nd gen upgrade.
    eventTrigger: Describes the Event trigger which has been setup to prepare
      for 2nd gen upgrade.
    serviceConfig: Describes the Cloud Run service which has been setup to
      prepare for 2nd gen upgrade.
    upgradeState: UpgradeState of the function
  """

    class UpgradeStateValueValuesEnum(_messages.Enum):
        """UpgradeState of the function

    Values:
      UPGRADE_STATE_UNSPECIFIED: Unspecified state. Most functions are in this
        upgrade state.
      ELIGIBLE_FOR_2ND_GEN_UPGRADE: Functions in this state are eligible for
        1st Gen -> 2nd Gen upgrade.
      UPGRADE_OPERATION_IN_PROGRESS: An upgrade related operation is in
        progress.
      SETUP_FUNCTION_UPGRADE_CONFIG_SUCCESSFUL: SetupFunctionUpgradeConfig API
        was successful and a 2nd Gen function has been created based on 1st
        Gen function instance.
      SETUP_FUNCTION_UPGRADE_CONFIG_ERROR: SetupFunctionUpgradeConfig API was
        un-successful.
      ABORT_FUNCTION_UPGRADE_ERROR: AbortFunctionUpgrade API was un-
        successful.
      REDIRECT_FUNCTION_UPGRADE_TRAFFIC_SUCCESSFUL:
        RedirectFunctionUpgradeTraffic API was successful and traffic is
        served by 2nd Gen function stack.
      REDIRECT_FUNCTION_UPGRADE_TRAFFIC_ERROR: RedirectFunctionUpgradeTraffic
        API was un-successful.
      ROLLBACK_FUNCTION_UPGRADE_TRAFFIC_ERROR: RollbackFunctionUpgradeTraffic
        API was un-successful.
      COMMIT_FUNCTION_UPGRADE_ERROR: CommitFunctionUpgrade API was un-
        successful.
    """
        UPGRADE_STATE_UNSPECIFIED = 0
        ELIGIBLE_FOR_2ND_GEN_UPGRADE = 1
        UPGRADE_OPERATION_IN_PROGRESS = 2
        SETUP_FUNCTION_UPGRADE_CONFIG_SUCCESSFUL = 3
        SETUP_FUNCTION_UPGRADE_CONFIG_ERROR = 4
        ABORT_FUNCTION_UPGRADE_ERROR = 5
        REDIRECT_FUNCTION_UPGRADE_TRAFFIC_SUCCESSFUL = 6
        REDIRECT_FUNCTION_UPGRADE_TRAFFIC_ERROR = 7
        ROLLBACK_FUNCTION_UPGRADE_TRAFFIC_ERROR = 8
        COMMIT_FUNCTION_UPGRADE_ERROR = 9
    buildConfig = _messages.MessageField('BuildConfig', 1)
    eventTrigger = _messages.MessageField('EventTrigger', 2)
    serviceConfig = _messages.MessageField('ServiceConfig', 3)
    upgradeState = _messages.EnumField('UpgradeStateValueValuesEnum', 4)