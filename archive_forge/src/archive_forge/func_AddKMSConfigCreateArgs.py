from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddKMSConfigCreateArgs(parser):
    """Add args for creating a KMS Config."""
    concept_parsers.ConceptParser([flags.GetKmsConfigPresentationSpec('The KMS Config to create')]).AddToParser(parser)
    AddKmsKeyResourceArg(parser, required=True)
    flags.AddResourceDescriptionArg(parser, 'KMS Config')
    flags.AddResourceAsyncFlag(parser)
    labels_util.AddCreateLabelsFlags(parser)