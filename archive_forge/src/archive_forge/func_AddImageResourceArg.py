from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute.images.packages import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddImageResourceArg(parser):
    """Add resource arg of image for 'packages list' command."""
    concept_parsers.ConceptParser([resource_args.CreateImageResourcePresentationSpec('Name of the disk image.')]).AddToParser(parser)