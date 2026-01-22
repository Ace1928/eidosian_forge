from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import googlecloudsdk
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def AddCreateTrigerResourceArgs(parser, release_track):
    """Adds trigger and channel arguments to for trigger creation."""
    if release_track == base.ReleaseTrack.GA:
        concept_parsers.ConceptParser([presentation_specs.ResourcePresentationSpec('trigger', TriggerResourceSpec(), 'The trigger to create.', required=True), presentation_specs.ResourcePresentationSpec('--channel', ChannelResourceSpec(), 'The channel to use in the trigger. The channel is needed only if trigger is created for a third-party provider.', flag_name_overrides={'location': ''})], command_level_fallthroughs={'--channel.location': ['trigger.location']}).AddToParser(parser)
    else:
        AddTriggerResourceArg(parser, 'The trigger to create.', required=True)