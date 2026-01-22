from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.notebooks import environments as env_util
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateSetAcceleratorRequest(args, messages):
    """Create and return Accelerator update request."""
    instance = GetInstanceResource(args).RelativeName()
    set_acc_request = messages.SetInstanceAcceleratorRequest()
    accelerator_config = messages.SetInstanceAcceleratorRequest
    if args.IsSpecified('accelerator_core_count'):
        set_acc_request.coreCount = args.accelerator_core_count
    if args.IsSpecified('accelerator_type'):
        type_enum = arg_utils.ChoiceEnumMapper(arg_name='accelerator-type', message_enum=accelerator_config.TypeValueValuesEnum, include_filter=lambda x: 'UNSPECIFIED' not in x).GetEnumForChoice(arg_utils.EnumNameToChoice(args.accelerator_type))
        set_acc_request.type = type_enum
    return messages.NotebooksProjectsLocationsInstancesSetAcceleratorRequest(name=instance, setInstanceAcceleratorRequest=set_acc_request)