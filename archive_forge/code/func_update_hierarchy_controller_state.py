from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet.config_management import utils
from googlecloudsdk.command_lib.container.fleet.features import base as feature_base
from googlecloudsdk.core import log
def update_hierarchy_controller_state(self, fs):
    """Update hierarchy controller state for the membership that has ACM installed.

    The PENDING state is set separately after this logic. The PENDING state
    suggests the HC part in feature_spec and feature_state are inconsistent, but
    the HC status from feature_state is not ERROR. This suggests that HC might
    be still in the updating process, so we mark it as PENDING

    Args:
      fs: ConfigmanagementFeatureState
    """
    if not (fs.hierarchyControllerState and fs.hierarchyControllerState.state):
        self.hierarchy_controller_state = NA
        return
    hc_deployment_state = fs.hierarchyControllerState.state
    hnc_state = 'NOT_INSTALLED'
    ext_state = 'NOT_INSTALLED'
    if hc_deployment_state.hnc:
        hnc_state = hc_deployment_state.hnc.name
    if hc_deployment_state.extension:
        ext_state = hc_deployment_state.extension.name
    deploys_to_status = {('INSTALLED', 'INSTALLED'): 'INSTALLED', ('INSTALLED', 'NOT_INSTALLED'): 'INSTALLED', ('NOT_INSTALLED', 'NOT_INSTALLED'): NA}
    if (hnc_state, ext_state) in deploys_to_status:
        self.hierarchy_controller_state = deploys_to_status[hnc_state, ext_state]
    else:
        self.hierarchy_controller_state = 'ERROR'