import argparse
from cliff import show
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0
class DeleteAutoAllocatedTopology(v2_0.NeutronCommand):
    """Delete the auto-allocated topology of a given tenant."""
    resource = 'auto_allocated_topology'

    def get_parser(self, prog_name):
        parser = super(DeleteAutoAllocatedTopology, self).get_parser(prog_name)
        parser.add_argument('--tenant-id', metavar='tenant-id', help=_('The owner tenant ID.'))
        parser.add_argument('pos_tenant_id', help=argparse.SUPPRESS, nargs='?')
        return parser

    def take_action(self, parsed_args):
        client = self.get_client()
        tenant_id = parsed_args.tenant_id or parsed_args.pos_tenant_id
        client.delete_auto_allocated_topology(tenant_id)
        tenant_id = tenant_id or 'None (i.e. yours)'
        print(_('Deleted topology for tenant %s.') % tenant_id, file=self.app.stdout)