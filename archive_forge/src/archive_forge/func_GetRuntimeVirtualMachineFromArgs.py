from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GetRuntimeVirtualMachineFromArgs():
    machine_type = 'n1-standard-4'
    if args.IsSpecified('machine_type'):
        machine_type = args.machine_type
    virtual_machine_config = messages.VirtualMachineConfig(machineType=machine_type, dataDisk=messages.LocalDisk())
    return messages.VirtualMachine(virtualMachineConfig=virtual_machine_config)