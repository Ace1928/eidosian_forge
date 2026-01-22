from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0.bgp import speaker as bgp_speaker
class ListBGPSpeakersOnDRAgent(neutronV20.ListCommand):
    """List BGP speakers hosted by a Dynamic Routing agent."""
    list_columns = ['id', 'name', 'local_as', 'ip_version']
    resource = 'bgp_speaker'

    def get_parser(self, prog_name):
        parser = super(ListBGPSpeakersOnDRAgent, self).get_parser(prog_name)
        parser.add_argument('dragent_id', metavar='BGP_DRAGENT_ID', help=_('ID of the Dynamic Routing agent.'))
        return parser

    def call_server(self, neutron_client, search_opts, parsed_args):
        data = neutron_client.list_bgp_speaker_on_dragent(parsed_args.dragent_id, **search_opts)
        return data