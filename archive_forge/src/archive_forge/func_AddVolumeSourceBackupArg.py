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
def AddVolumeSourceBackupArg(parser):
    """Adds the --source-backup arg to the arg parser."""
    concept_parsers.ConceptParser.ForResource('--source-backup', flags.GetBackupResourceSpec(positional=False), flag_name_overrides={'location': ''}, group_help='The source Backup to create the Volume from.').AddToParser(parser)