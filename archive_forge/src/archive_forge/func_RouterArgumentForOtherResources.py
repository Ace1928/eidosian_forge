from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def RouterArgumentForOtherResources(required=True, suppress_region=True):
    region_explanation = 'Should be the same as --region, if not specified, it will be inherited from --region.'
    return compute_flags.ResourceArgument(resource_name='router', name='--router', completer=RoutersCompleter, plural=False, required=required, regional_collection='compute.routers', short_help='The Google Cloud Router to use for dynamic routing.', region_explanation=region_explanation, region_hidden=suppress_region)