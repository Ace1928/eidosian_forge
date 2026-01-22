import logging
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.services.protocols.bgp.info_base.base import Destination
from os_ken.services.protocols.bgp.info_base.base import NonVrfPathProcessingMixin
from os_ken.services.protocols.bgp.info_base.base import Path
from os_ken.services.protocols.bgp.info_base.base import Table
class RtcDest(Destination, NonVrfPathProcessingMixin):
    ROUTE_FAMILY = RF_RTC_UC

    def _new_best_path(self, new_best_path):
        NonVrfPathProcessingMixin._new_best_path(self, new_best_path)

    def _best_path_lost(self):
        NonVrfPathProcessingMixin._best_path_lost(self)