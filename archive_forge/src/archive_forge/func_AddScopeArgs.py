from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as mig_flags
from googlecloudsdk.command_lib.util import completers
import six
def AddScopeArgs(parser, multizonal):
    """Adds flags for group scope."""
    if multizonal:
        scope_parser = parser.add_mutually_exclusive_group()
        flags.AddRegionFlag(scope_parser, resource_type='instance group', operation_type='set named ports for', explanation=flags.REGION_PROPERTY_EXPLANATION_NO_DEFAULT)
        flags.AddZoneFlag(scope_parser, resource_type='instance group', operation_type='set named ports for', explanation=flags.ZONE_PROPERTY_EXPLANATION_NO_DEFAULT)
    else:
        flags.AddZoneFlag(parser, resource_type='instance group', operation_type='set named ports for')