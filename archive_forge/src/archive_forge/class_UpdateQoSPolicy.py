import os
from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronv20
class UpdateQoSPolicy(neutronv20.UpdateCommand):
    """Update a given qos policy."""
    resource = 'policy'
    shadow_resource = 'qos_policy'

    def add_known_arguments(self, parser):
        parser.add_argument('--name', help=_('Name of the QoS policy.'))
        parser.add_argument('--description', help=_('Description of the QoS policy.'))
        shared_group = parser.add_mutually_exclusive_group()
        shared_group.add_argument('--shared', action='store_true', help=_('Accessible by other tenants. Set shared to True (default is False).'))
        shared_group.add_argument('--no-shared', action='store_true', help=_('Not accessible by other tenants. Set shared to False.'))

    def args2body(self, parsed_args):
        body = {}
        if parsed_args.name:
            body['name'] = parsed_args.name
        if parsed_args.description:
            body['description'] = parsed_args.description
        if parsed_args.shared:
            body['shared'] = True
        if parsed_args.no_shared:
            body['shared'] = False
        return {self.resource: body}