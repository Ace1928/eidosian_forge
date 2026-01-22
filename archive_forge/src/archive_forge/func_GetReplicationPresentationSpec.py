from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def GetReplicationPresentationSpec(group_help):
    return presentation_specs.ResourcePresentationSpec('replication', GetReplicationResourceSpec(), group_help, required=True, flag_name_overrides={'volume': ''})