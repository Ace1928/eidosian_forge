from osc_lib.command import command
from osc_lib import utils
from neutronclient._i18n import _
from neutronclient.osc import utils as nc_osc_utils
from neutronclient.osc.v2.dynamic_routing import constants
class SetBgpSpeaker(command.Command):
    _description = _('Set BGP speaker properties')
    resource = constants.BGP_SPEAKER

    def get_parser(self, prog_name):
        parser = super(SetBgpSpeaker, self).get_parser(prog_name)
        parser.add_argument('bgp_speaker', metavar='<bgp-speaker>', help=_('BGP speaker to update (name or ID)'))
        parser.add_argument('--name', help=_('New name for the BGP speaker'))
        add_common_arguments(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        id = client.find_bgp_speaker(parsed_args.bgp_speaker)['id']
        attrs = _get_attrs(self.app.client_manager, parsed_args)
        client.update_bgp_speaker(id, **attrs)