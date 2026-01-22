from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronv20
class UpdateEndpointGroup(neutronv20.UpdateCommand):
    """Update a given VPN endpoint group."""
    resource = 'endpoint_group'

    def add_known_arguments(self, parser):
        add_known_endpoint_group_arguments(parser, is_create=False)

    def args2body(self, parsed_args):
        body = {}
        neutronv20.update_dict(parsed_args, body, ['name', 'description'])
        return {self.resource: body}