from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class UpdateL7Policy(neutronV20.UpdateCommand):
    """LBaaS v2 Update a given L7 policy."""
    resource = 'l7policy'
    shadow_resource = 'lbaas_l7policy'

    def add_known_arguments(self, parser):
        _add_common_args(parser, is_create=False)
        utils.add_boolean_argument(parser, '--admin-state-up', help=_('Specify the administrative state of the policy (True meaning "Up").'))

    def args2body(self, parsed_args):
        return _common_args2body(self.get_client(), parsed_args, False)