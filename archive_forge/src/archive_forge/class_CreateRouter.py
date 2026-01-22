import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import availability_zone
class CreateRouter(neutronV20.CreateCommand):
    """Create a router for a given tenant."""
    resource = 'router'
    _formatters = {'external_gateway_info': _format_external_gateway_info}

    def add_known_arguments(self, parser):
        parser.add_argument('--admin-state-down', dest='admin_state', action='store_false', help=_('Set admin state up to false.'))
        parser.add_argument('--admin_state_down', dest='admin_state', action='store_false', help=argparse.SUPPRESS)
        parser.add_argument('name', metavar='NAME', help=_('Name of the router to be created.'))
        parser.add_argument('--description', help=_('Description of router.'))
        parser.add_argument('--flavor', help=_('ID or name of flavor.'))
        utils.add_boolean_argument(parser, '--distributed', dest='distributed', help=_('Create a distributed router.'))
        utils.add_boolean_argument(parser, '--ha', dest='ha', help=_('Create a highly available router.'))
        availability_zone.add_az_hint_argument(parser, self.resource)

    def args2body(self, parsed_args):
        body = {'admin_state_up': parsed_args.admin_state}
        if parsed_args.flavor:
            _flavor_id = neutronV20.find_resourceid_by_name_or_id(self.get_client(), 'flavor', parsed_args.flavor)
            body['flavor_id'] = _flavor_id
        neutronV20.update_dict(parsed_args, body, ['name', 'tenant_id', 'distributed', 'ha', 'description'])
        availability_zone.args2body_az_hint(parsed_args, body)
        return {self.resource: body}