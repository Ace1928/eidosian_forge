from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class LbaasMemberMixin(object):

    def set_extra_attrs(self, parsed_args):
        self.parent_id = _get_pool_id(self.get_client(), parsed_args.pool)

    def add_known_arguments(self, parser):
        parser.add_argument('pool', metavar='POOL', help=_('ID or name of the pool that this member belongs to.'))