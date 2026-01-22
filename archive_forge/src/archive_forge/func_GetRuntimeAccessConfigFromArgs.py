from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GetRuntimeAccessConfigFromArgs():
    runtime_access_config = messages.RuntimeAccessConfig
    type_enum = None
    if args.IsSpecified('runtime_access_type'):
        type_enum = arg_utils.ChoiceEnumMapper(arg_name='runtime-access-type', message_enum=runtime_access_config.AccessTypeValueValuesEnum, include_filter=lambda x: 'UNSPECIFIED' not in x).GetEnumForChoice(arg_utils.EnumNameToChoice(args.runtime_access_type))
    return runtime_access_config(accessType=type_enum, runtimeOwner=args.runtime_owner)