import logging
from blazarclient import command
class ListFloatingIPs(command.ListCommand):
    """Print a list of floating IPs."""
    resource = 'floatingip'
    log = logging.getLogger(__name__ + '.ListFloatingIPs')
    list_columns = ['id', 'floating_ip_address', 'floating_network_id']

    def get_parser(self, prog_name):
        parser = super(ListFloatingIPs, self).get_parser(prog_name)
        parser.add_argument('--sort-by', metavar='<floatingip_column>', help='column name used to sort result', default='id')
        return parser