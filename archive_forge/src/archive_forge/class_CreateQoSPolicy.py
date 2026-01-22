import os
from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronv20
class CreateQoSPolicy(neutronv20.CreateCommand):
    """Create a qos policy."""
    resource = 'policy'
    shadow_resource = 'qos_policy'

    def add_known_arguments(self, parser):
        parser.add_argument('name', metavar='NAME', help=_('Name of the QoS policy to be created.'))
        parser.add_argument('--description', help=_('Description of the QoS policy to be created.'))
        parser.add_argument('--shared', action='store_true', help=_('Accessible by other tenants. Set shared to True (default is False).'))

    def args2body(self, parsed_args):
        body = {'name': parsed_args.name}
        if parsed_args.description:
            body['description'] = parsed_args.description
        if parsed_args.shared:
            body['shared'] = parsed_args.shared
        if parsed_args.tenant_id:
            body['tenant_id'] = parsed_args.tenant_id
        return {self.resource: body}