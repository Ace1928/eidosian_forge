from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.core import log
def AddInstanceGroup(parser, operation_type, with_deprecated_zone=False):
    """Adds arguments to define instance group."""
    parser.add_argument('--instance-group', required=True, help='The name or URI of a Google Cloud Instance Group.')
    scope_parser = parser.add_mutually_exclusive_group()
    flags.AddRegionFlag(scope_parser, resource_type='instance group', operation_type='{0} the backend service'.format(operation_type), flag_prefix='instance-group', explanation=flags.REGION_PROPERTY_EXPLANATION_NO_DEFAULT)
    if with_deprecated_zone:
        flags.AddZoneFlag(scope_parser, resource_type='instance group', operation_type='{0} the backend service'.format(operation_type), explanation='DEPRECATED, use --instance-group-zone flag instead.')
    flags.AddZoneFlag(scope_parser, resource_type='instance group', operation_type='{0} the backend service'.format(operation_type), flag_prefix='instance-group', explanation=flags.ZONE_PROPERTY_EXPLANATION_NO_DEFAULT)