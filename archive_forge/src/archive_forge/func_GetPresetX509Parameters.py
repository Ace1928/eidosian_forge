from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import exceptions
def GetPresetX509Parameters(profile_name):
    """Parses the profile name string into the corresponding API X509Parameters.

  Args:
    profile_name: The preset profile name.

  Returns:
    An X509Parameters object.
  """
    if profile_name not in _PRESET_PROFILES:
        raise exceptions.InvalidArgumentException('--use-preset-profile', 'The preset profile that was specified does not exist.')
    messages = privateca_base.GetMessagesModule('v1')
    return messages_util.DictToMessageWithErrorCheck(_PRESET_PROFILES[profile_name], messages.X509Parameters)