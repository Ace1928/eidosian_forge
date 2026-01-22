from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class UpdateHealthMonitor(neutronV20.UpdateCommand):
    """LBaaS v2 Update a given healthmonitor."""
    resource = 'healthmonitor'
    shadow_resource = 'lbaas_healthmonitor'

    def add_known_arguments(self, parser):
        _add_common_args(parser, is_create=False)
        utils.add_boolean_argument(parser, '--admin-state-up', help=_('Update the administrative state of the health monitor (True meaning "Up").'))

    def args2body(self, parsed_args):
        body = {}
        _parse_common_args(body, parsed_args)
        neutronV20.update_dict(parsed_args, body, ['admin_state_up'])
        return {self.resource: body}