from osc_lib.command import command
from osc_lib import utils
from neutronclient._i18n import _
class ListDRAgent(command.Lister):
    """List dynamic routing agents"""
    resource = 'agent'
    list_columns = ['id', 'host', 'admin_state_up', 'alive']
    unknown_parts_flag = False

    def get_parser(self, prog_name):
        parser = super(ListDRAgent, self).get_parser(prog_name)
        parser.add_argument('--bgp-speaker', metavar='<bgp-speaker>', help=_('List dynamic routing agents hosting a BGP speaker (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        if parsed_args.bgp_speaker is not None:
            speaker_id = client.find_bgp_speaker(parsed_args.bgp_speaker).id
            data = client.get_bgp_dragents_hosting_speaker(speaker_id)
        else:
            attrs = {'agent_type': 'BGP dynamic routing agent'}
            data = client.agents(**attrs)
        columns = ('id', 'agent_type', 'host', 'availability_zone', 'is_alive', 'is_admin_state_up', 'binary')
        column_headers = ('ID', 'Agent Type', 'Host', 'Availability Zone', 'Alive', 'State', 'Binary')
        return (column_headers, (utils.get_item_properties(s, columns) for s in data))