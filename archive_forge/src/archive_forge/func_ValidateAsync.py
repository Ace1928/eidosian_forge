from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container.fleet.scopes.rollout_sequencing import base
from googlecloudsdk.core import log
def ValidateAsync(ref, args, request):
    del ref
    cmd = base.ClusterUpgradeCommand(args)
    is_async = args.IsKnownAndSpecified('async_') and args.async_
    if cmd.IsClusterUpgradeRequest() and is_async:
        raise exceptions.ConflictingArgumentsException('--async cannot be specified with Rollout Sequencing flags')
    return request