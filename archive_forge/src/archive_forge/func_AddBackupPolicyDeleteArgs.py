from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddBackupPolicyDeleteArgs(parser):
    """Add args for deleting a Backup Policy."""
    concept_parsers.ConceptParser([flags.GetBackupPolicyPresentationSpec('The Backup Policy to delete')]).AddToParser(parser)
    flags.AddResourceAsyncFlag(parser)