from osc_lib.command import command
from osc_lib import utils
from neutronclient._i18n import _
class AddBgpSpeakerToDRAgent(command.Command):
    """Add a BGP speaker to a dynamic routing agent"""

    def get_parser(self, prog_name):
        parser = super(AddBgpSpeakerToDRAgent, self).get_parser(prog_name)
        add_common_args(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        speaker_id = client.find_bgp_speaker(parsed_args.bgp_speaker).id
        client.add_bgp_speaker_to_dragent(parsed_args.dragent_id, speaker_id)