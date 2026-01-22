from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GetPostStartupScriptBehavior():
    type_enum = None
    if args.IsSpecified('post_startup_script_behavior'):
        runtime_software_config_message = messages.RuntimeSoftwareConfig
        type_enum = arg_utils.ChoiceEnumMapper(arg_name='post-startup-script-behavior', message_enum=runtime_software_config_message.PostStartupScriptBehaviorTypeValueValuesEnum, include_filter=lambda x: 'UNSPECIFIED' not in x).GetEnumForChoice(arg_utils.EnumNameToChoice(args.post_startup_script_behavior))
    return type_enum