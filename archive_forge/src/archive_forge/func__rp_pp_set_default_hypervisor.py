import uuid
import os_traits
from neutron_lib._i18n import _
from neutron_lib import constants as const
from neutron_lib.placement import constants as place_const
def _rp_pp_set_default_hypervisor(cfg, host):
    if cfg.get('') and cfg.get(host):
        raise ValueError(_('Found configuration for "%s" hypervisor and one without hypervisor name specified that would override it.') % host)
    if cfg.get(''):
        cfg[host] = cfg.pop('')