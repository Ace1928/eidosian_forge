from osc_lib.command import command
from osc_lib import utils
from neutronclient._i18n import _
from neutronclient.osc import utils as nc_osc_utils
from neutronclient.osc.v2.dynamic_routing import constants
class DeleteBgpSpeaker(command.Command):
    _description = _('Delete a BGP speaker')

    def get_parser(self, prog_name):
        parser = super(DeleteBgpSpeaker, self).get_parser(prog_name)
        parser.add_argument('bgp_speaker', metavar='<bgp-speaker>', help=_('BGP speaker to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        id = client.find_bgp_speaker(parsed_args.bgp_speaker)['id']
        client.delete_bgp_speaker(id)