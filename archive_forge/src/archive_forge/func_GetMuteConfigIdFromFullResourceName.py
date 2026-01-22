from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util as scc_util
def GetMuteConfigIdFromFullResourceName(mute_config):
    """Gets muteConfig id from the full resource name."""
    mute_config_components = mute_config.split('/')
    return mute_config_components[len(mute_config_components) - 1]