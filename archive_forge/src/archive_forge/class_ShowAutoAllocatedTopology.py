import argparse
from cliff import show
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0
class ShowAutoAllocatedTopology(v2_0.NeutronCommand, show.ShowOne):
    """Show the auto-allocated topology of a given tenant."""
    resource = 'auto_allocated_topology'

    def get_parser(self, prog_name):
        parser = super(ShowAutoAllocatedTopology, self).get_parser(prog_name)
        parser.add_argument('--dry-run', help=_('Validate the requirements for auto-allocated-topology. (Does not return a topology.)'), action='store_true')
        parser.add_argument('--tenant-id', metavar='tenant-id', help=_('The owner tenant ID.'))
        parser.add_argument('pos_tenant_id', help=argparse.SUPPRESS, nargs='?')
        return parser

    def take_action(self, parsed_args):
        client = self.get_client()
        extra_values = v2_0.parse_args_to_dict(self.values_specs)
        if extra_values:
            raise exceptions.CommandError(_('Invalid argument(s): --%s') % ', --'.join(extra_values))
        tenant_id = parsed_args.tenant_id or parsed_args.pos_tenant_id
        if parsed_args.dry_run:
            data = client.validate_auto_allocated_topology_requirements(tenant_id)
        else:
            data = client.get_auto_allocated_topology(tenant_id)
        if self.resource in data:
            for k, v in data[self.resource].items():
                if isinstance(v, list):
                    value = ''
                    for _item in v:
                        if value:
                            value += '\n'
                        if isinstance(_item, dict):
                            value += jsonutils.dumps(_item)
                        else:
                            value += str(_item)
                    data[self.resource][k] = value
                elif v == 'dry-run=pass':
                    return (('dry-run',), ('pass',))
                elif v is None:
                    data[self.resource][k] = ''
            return zip(*sorted(data[self.resource].items()))
        else:
            return None