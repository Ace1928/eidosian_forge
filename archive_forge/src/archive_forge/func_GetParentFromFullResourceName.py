from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util as scc_util
def GetParentFromFullResourceName(mute_config, version):
    """Gets parent from the full resource name."""
    mute_config_components = mute_config.split('/')
    if version == 'v1':
        return f'{mute_config_components[0]}/{mute_config_components[1]}'
    if version == 'v2':
        return f'{mute_config_components[0]}/{mute_config_components[1]}/{mute_config_components[2]}/{mute_config_components[3]}'