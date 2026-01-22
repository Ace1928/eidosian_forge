from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class UpdateLoadBalancer(neutronV20.UpdateCommand):
    """LBaaS v2 Update a given loadbalancer."""
    resource = 'loadbalancer'

    def add_known_arguments(self, parser):
        utils.add_boolean_argument(parser, '--admin-state-up', help=_('Update the administrative state of the load balancer (True meaning "Up").'))
        _add_common_args(parser)

    def args2body(self, parsed_args):
        body = {}
        _parse_common_args(body, parsed_args)
        neutronV20.update_dict(parsed_args, body, ['admin_state_up'])
        return {self.resource: body}