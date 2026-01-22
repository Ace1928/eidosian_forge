import json
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.resource_manager import folders
from googlecloudsdk.command_lib.scc.manage import constants
from googlecloudsdk.command_lib.scc.manage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.generated_clients.apis.securitycentermanagement.v1 import securitycentermanagement_v1_messages as messages
def GetEnablementStateFromArgs(enablement_state: str, module_type: constants.CustomModuleType):
    """Parse the enablement state."""
    if module_type == constants.CustomModuleType.SHA:
        state_enum = messages.SecurityHealthAnalyticsCustomModule.EnablementStateValueValuesEnum
    elif module_type == constants.CustomModuleType.ETD:
        state_enum = messages.EventThreatDetectionCustomModule.EnablementStateValueValuesEnum
    else:
        raise errors.InvalidModuleTypeError(f'Module type "{module_type}" is not a valid module type.')
    if enablement_state is None:
        raise errors.InvalidEnablementStateError('Error parsing enablement state. Enablement state cannot be empty.')
    state = enablement_state.upper()
    if state == 'ENABLED':
        return state_enum.ENABLED
    elif state == 'DISABLED':
        return state_enum.DISABLED
    elif state == 'INHERITED':
        return state_enum.INHERITED
    else:
        raise errors.InvalidEnablementStateError(f'Error parsing enablement state. "{state}" is not a valid enablement state. Please provide one of ENABLED, DISABLED, or INHERITED.')