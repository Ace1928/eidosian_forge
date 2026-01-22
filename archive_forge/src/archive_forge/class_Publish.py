from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.eventarc import channels
from googlecloudsdk.api_lib.eventarc import common_publishing
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.eventarc import flags
from googlecloudsdk.core import log
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Publish(base.Command):
    """Publish to an Eventarc channel."""
    detailed_help = _DETAILED_HELP

    @classmethod
    def Args(cls, parser):
        flags.AddChannelResourceArg(parser, 'Channel to Publish to.', required=True)
        flags.AddEventPublishingArgs(parser)

    def Run(self, args):
        """Run the Publish command."""
        client = channels.ChannelClientV1()
        channel_ref = args.CONCEPTS.channel.Parse()
        name = channel_ref.Name()
        log.debug('Publishing event with id: {} to channel: {}'.format(args.event_id, name))
        client.Publish(channel_ref, common_publishing.CreateCloudEvent(args.event_id, args.event_type, args.event_source, args.event_data, args.event_attributes))
        return log.out.Print('Event published successfully')