from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddVolumeSourceSnapshotArg(parser):
    """Adds the --source-snapshot arg to the arg parser."""
    concept_parsers.ConceptParser.ForResource('--source-snapshot', flags.GetSnapshotResourceSpec(source_snapshot_op=True, positional=False), flag_name_overrides={'location': '', 'volume': ''}, group_help='The source Snapshot to create the Volume from.').AddToParser(parser)