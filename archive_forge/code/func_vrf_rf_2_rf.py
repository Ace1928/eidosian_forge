import abc
import logging
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.lib.packet.bgp import RF_IPv6_UC
from os_ken.lib.packet.bgp import RF_L2_EVPN
from os_ken.lib.packet.bgp import RF_IPv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_IPv6_FLOWSPEC
from os_ken.lib.packet.bgp import RF_L2VPN_FLOWSPEC
from os_ken.services.protocols.bgp.utils import validation
from os_ken.services.protocols.bgp.base import get_validator
from os_ken.services.protocols.bgp.rtconf.base import BaseConf
from os_ken.services.protocols.bgp.rtconf.base import BaseConfListener
from os_ken.services.protocols.bgp.rtconf.base import ConfigTypeError
from os_ken.services.protocols.bgp.rtconf.base import ConfigValueError
from os_ken.services.protocols.bgp.rtconf.base import ConfWithId
from os_ken.services.protocols.bgp.rtconf.base import ConfWithIdListener
from os_ken.services.protocols.bgp.rtconf.base import ConfWithStats
from os_ken.services.protocols.bgp.rtconf.base import ConfWithStatsListener
from os_ken.services.protocols.bgp.rtconf.base import MAX_NUM_EXPORT_RT
from os_ken.services.protocols.bgp.rtconf.base import MAX_NUM_IMPORT_RT
from os_ken.services.protocols.bgp.rtconf.base import MULTI_EXIT_DISC
from os_ken.services.protocols.bgp.rtconf.base import RuntimeConfigError
from os_ken.services.protocols.bgp.rtconf.base import SITE_OF_ORIGINS
from os_ken.services.protocols.bgp.rtconf.base import validate
from os_ken.services.protocols.bgp.rtconf.base import validate_med
from os_ken.services.protocols.bgp.rtconf.base import validate_soo_list
@staticmethod
def vrf_rf_2_rf(vrf_rf):
    if vrf_rf == VRF_RF_IPV4:
        return RF_IPv4_UC
    elif vrf_rf == VRF_RF_IPV6:
        return RF_IPv6_UC
    elif vrf_rf == VRF_RF_L2_EVPN:
        return RF_L2_EVPN
    elif vrf_rf == VRF_RF_IPV4_FLOWSPEC:
        return RF_IPv4_FLOWSPEC
    elif vrf_rf == VRF_RF_IPV6_FLOWSPEC:
        return RF_IPv6_FLOWSPEC
    elif vrf_rf == VRF_RF_L2VPN_FLOWSPEC:
        return RF_L2VPN_FLOWSPEC
    else:
        raise ValueError('Unsupported VRF route family given %s' % vrf_rf)