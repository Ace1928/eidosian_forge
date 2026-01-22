from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class UpdateL7Rule(LbaasL7RuleMixin, neutronV20.UpdateCommand):
    """LBaaS v2 Update a given L7 rule."""
    resource = 'rule'
    shadow_resource = 'lbaas_l7rule'

    def add_known_arguments(self, parser):
        super(UpdateL7Rule, self).add_known_arguments(parser)
        _add_common_args(parser, False)
        utils.add_boolean_argument(parser, '--admin-state-up', help=_('Specify the administrative state of the rule (True meaning "Up").'))

    def args2body(self, parsed_args):
        return _common_args2body(self.get_client(), parsed_args, False)