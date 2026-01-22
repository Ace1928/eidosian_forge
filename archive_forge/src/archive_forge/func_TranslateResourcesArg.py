from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
from googlecloudsdk.command_lib.compute.reservations import resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
def TranslateResourcesArg(messages, resources_arg):
    return [messages.ResourceCommitment(amount=resources_arg['vcpu'], type=messages.ResourceCommitment.TypeValueValuesEnum.VCPU), messages.ResourceCommitment(amount=resources_arg['memory'] // (1024 * 1024), type=messages.ResourceCommitment.TypeValueValuesEnum.MEMORY)]