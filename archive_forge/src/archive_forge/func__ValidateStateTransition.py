from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.functions import api_enablement
from googlecloudsdk.api_lib.functions.v2 import client as client_v2
from googlecloudsdk.api_lib.functions.v2 import exceptions
from googlecloudsdk.api_lib.functions.v2 import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.eventarc import types as trigger_types
from googlecloudsdk.command_lib.functions import flags
from googlecloudsdk.command_lib.functions.v2 import deploy_util
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
import six
def _ValidateStateTransition(upgrade_state, action):
    """Validates whether the action is a valid action for the given upgrade state."""
    upgrade_state_str = six.text_type(upgrade_state)
    if upgrade_state_str == 'UPGRADE_OPERATION_IN_PROGRESS':
        raise exceptions.FunctionsError('An upgrade operation is already in progress for this function. Please try again later.')
    if upgrade_state_str == action.target_state:
        raise exceptions.FunctionsError('This function is already in the desired upgrade state: {}'.format(upgrade_state))
    if action not in _VALID_TRANSITION_ACTIONS[upgrade_state_str]:
        raise exceptions.FunctionsError("This function is not eligible for this operation. Its current upgrade state is '{}'.".format(upgrade_state))