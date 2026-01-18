import logging
from os_ken.lib.packet.bgp import FlowSpecIPv4NLRI
from os_ken.lib.packet.bgp import RF_IPv4_FLOWSPEC
from os_ken.services.protocols.bgp.info_base.base import Path
from os_ken.services.protocols.bgp.info_base.base import Table
from os_ken.services.protocols.bgp.info_base.base import Destination
from os_ken.services.protocols.bgp.info_base.base import NonVrfPathProcessingMixin
Return a key that will uniquely identify this NLRI inside
        this table.
        