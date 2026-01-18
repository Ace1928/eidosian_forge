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
Removes any matching `VrfConf` for given `route_dist` or `vrf_id`

        Parameters:
            - `route_dist`: (str) route distinguisher of a configured VRF
            - `vrf_id`: (str) vrf ID
            - `vrf_rf`: (str) route family of the VRF configuration
        If only `route_dist` is given, removes `VrfConf`s for all supported
        address families for this `route_dist`. If `vrf_rf` is given, than only
        removes `VrfConf` for that specific route family. If only `vrf_id` is
        given, matching `VrfConf` will be removed.
        