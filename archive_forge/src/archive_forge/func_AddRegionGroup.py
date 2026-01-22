from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.network_connectivity import util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddRegionGroup(parser, supports_region_wildcard=False, hide_global_arg=False, hide_region_arg=False):
    """Add a group which contains the global and region arguments to the given parser."""
    region_group = parser.add_group(required=False, mutex=True)
    AddGlobalFlag(region_group, hide_global_arg)
    AddRegionFlag(region_group, supports_region_wildcard, hide_region_arg)