from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddReplicationVolumeArg(parser, reverse_op=False):
    group_help = 'The Volume that the Replication is based on' if not reverse_op else 'The source Volume to reverse the Replication direction of'
    concept_parsers.ConceptParser.ForResource('--volume', flags.GetVolumeResourceSpec(positional=False), group_help=group_help, flag_name_overrides={'location': ''}).AddToParser(parser)