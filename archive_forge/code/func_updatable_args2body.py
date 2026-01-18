from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def updatable_args2body(parsed_args, body):
    neutronV20.update_dict(parsed_args, body, ['name', 'prefixes', 'default_prefixlen', 'min_prefixlen', 'max_prefixlen', 'is_default', 'description'])