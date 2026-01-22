from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateRuntime(args, messages):
    """Creates the Runtime message for the create request.

  Args:
    args: Argparse object from Command.Run
    messages: Module containing messages definition for the specified API.

  Returns:
    Runtime of the Runtime message.
  """

    def GetRuntimeVirtualMachineFromArgs():
        machine_type = 'n1-standard-4'
        if args.IsSpecified('machine_type'):
            machine_type = args.machine_type
        virtual_machine_config = messages.VirtualMachineConfig(machineType=machine_type, dataDisk=messages.LocalDisk())
        return messages.VirtualMachine(virtualMachineConfig=virtual_machine_config)

    def GetRuntimeAccessConfigFromArgs():
        runtime_access_config = messages.RuntimeAccessConfig
        type_enum = None
        if args.IsSpecified('runtime_access_type'):
            type_enum = arg_utils.ChoiceEnumMapper(arg_name='runtime-access-type', message_enum=runtime_access_config.AccessTypeValueValuesEnum, include_filter=lambda x: 'UNSPECIFIED' not in x).GetEnumForChoice(arg_utils.EnumNameToChoice(args.runtime_access_type))
        return runtime_access_config(accessType=type_enum, runtimeOwner=args.runtime_owner)

    def GetPostStartupScriptBehavior():
        type_enum = None
        if args.IsSpecified('post_startup_script_behavior'):
            runtime_software_config_message = messages.RuntimeSoftwareConfig
            type_enum = arg_utils.ChoiceEnumMapper(arg_name='post-startup-script-behavior', message_enum=runtime_software_config_message.PostStartupScriptBehaviorTypeValueValuesEnum, include_filter=lambda x: 'UNSPECIFIED' not in x).GetEnumForChoice(arg_utils.EnumNameToChoice(args.post_startup_script_behavior))
        return type_enum

    def GetRuntimeSoftwareConfigFromArgs():
        runtime_software_config = messages.RuntimeSoftwareConfig()
        if args.IsSpecified('idle_shutdown_timeout'):
            runtime_software_config.idleShutdownTimeout = args.idle_shutdown_timeout
        if args.IsSpecified('install_gpu_driver'):
            runtime_software_config.installGpuDriver = args.install_gpu_driver
        if args.IsSpecified('custom_gpu_driver_path'):
            runtime_software_config.customGpuDriverPath = args.custom_gpu_driver_path
        if args.IsSpecified('post_startup_script'):
            runtime_software_config.postStartupScript = args.post_startup_script
        if args.IsSpecified('post_startup_script_behavior'):
            runtime_software_config.postStartupScriptBehavior = GetPostStartupScriptBehavior()
        return runtime_software_config
    runtime = messages.Runtime(name=args.runtime, virtualMachine=GetRuntimeVirtualMachineFromArgs(), accessConfig=GetRuntimeAccessConfigFromArgs(), softwareConfig=GetRuntimeSoftwareConfigFromArgs())
    return runtime