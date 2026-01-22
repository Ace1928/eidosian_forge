from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddBackupPolicyUpdateArgs(parser):
    """Add args for updating a Backup Policy."""
    concept_parsers.ConceptParser([flags.GetBackupPolicyPresentationSpec('The Backup Policy to update')]).AddToParser(parser)
    AddBackupPolicyEnabledArg(parser)
    AddBackupPolicyBackupLimitGroup(parser)
    flags.AddResourceDescriptionArg(parser, 'Backup Policy')
    flags.AddResourceAsyncFlag(parser)
    labels_util.AddUpdateLabelsFlags(parser)