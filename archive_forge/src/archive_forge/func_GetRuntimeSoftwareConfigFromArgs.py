from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
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