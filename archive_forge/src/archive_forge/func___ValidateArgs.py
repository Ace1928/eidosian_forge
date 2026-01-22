from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.immersive_stream.xr import api_util
from googlecloudsdk.api_lib.immersive_stream.xr import instances
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.immersive_stream.xr import flags
from googlecloudsdk.command_lib.immersive_stream.xr import resource_args
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
@staticmethod
def __ValidateArgs(args):
    if args.add_region:
        return flags.ValidateRegionConfigArgs(args.add_region, 'add')
    if args.remove_region:
        if len(set(args.remove_region)) < len(args.remove_region):
            log.error('Duplicate regions in --remove-region arguments.')
            return False
    if args.update_region:
        return flags.ValidateRegionConfigArgs(args.update_region, 'update')
    return True