from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Dict, Union
from googlecloudsdk.calliope import parser_extensions
def GetFleetUpdateMask(args):
    fleet_flag_to_update_mask_paths = FlagToUpdateMaskPaths(FLEET_MESSAGE_TO_FLAGS)
    return GetUpdateMask(args, fleet_flag_to_update_mask_paths)