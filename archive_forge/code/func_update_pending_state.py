from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet.config_management import utils
from googlecloudsdk.command_lib.container.fleet.features import base as feature_base
from googlecloudsdk.core import log
def update_pending_state(self, feature_spec_mc, feature_state_mc):
    """Update config sync and policy controller with the pending state.

    Args:
      feature_spec_mc: MembershipConfig
      feature_state_mc: MembershipConfig
    """
    feature_state_pending = feature_state_mc is None and feature_spec_mc is not None
    if feature_state_pending:
        self.last_synced_token = 'PENDING'
        self.last_synced = 'PENDING'
        self.sync_branch = 'PENDING'
    if self.config_sync.__str__() in ['SYNCED', 'NOT_CONFIGURED', 'NOT_INSTALLED', NA] and (feature_state_pending or feature_spec_mc.configSync != feature_state_mc.configSync):
        self.config_sync = 'PENDING'
    if self.policy_controller_state.__str__() in ['INSTALLED', 'GatekeeperAudit NOT_INSTALLED', NA] and feature_state_pending:
        self.policy_controller_state = 'PENDING'
    if self.hierarchy_controller_state.__str__() != 'ERROR' and feature_state_pending or feature_spec_mc.hierarchyController != feature_state_mc.hierarchyController:
        self.hierarchy_controller_state = 'PENDING'