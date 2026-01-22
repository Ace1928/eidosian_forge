from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def MakeMachineImageArg(plural=False):
    return compute_flags.ResourceArgument(name='IMAGE', resource_name='machineImage', completer=compute_completers.MachineImagesCompleter, plural=plural, global_collection='compute.machineImages')