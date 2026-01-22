from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def _format_prefixes(subnetpool):
    try:
        return '\n'.join((pool for pool in subnetpool['prefixes']))
    except (TypeError, KeyError):
        return subnetpool['prefixes']